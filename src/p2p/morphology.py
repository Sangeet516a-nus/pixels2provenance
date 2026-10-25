"""Morphology measurement shared by Stage 1e (Pass 1) and Stage 5b (Pass 2).

statmorph (Rodriguez-Gomez et al. 2019) measures, for one source:
  non-parametric : Gini, M20, concentration C, asymmetry A, smoothness S,
                   multimode / intensity / deviation (MID), shape asymmetry
  parametric     : single Sersic fit (n, r_eff, ellipticity), convolved with
                   the PSF passed in -> this is where Pass 1 and Pass 2 differ.
If statmorph is missing or fails, simple NumPy estimators are used (flag = 4).
Masked pixels: other sources (segmentation stamp value 2) and cosmic-ray pixels
(value 3, Stage 1c). statmorph's -99 failure codes are stored as NaN, and a Sersic
fit that ends at a parameter limit (morphology.sersic_n_bounds,
sersic_rhalf_max_px) gets flag_sersic = 1.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import fftconvolve

try:
    import statmorph
    _HAVE_STATMORPH = True
except Exception:
    _HAVE_STATMORPH = False

FEATURES = ["gini", "m20", "concentration", "asymmetry", "smoothness", "sersic_n", "sersic_rhalf",
            "sersic_ellip", "r_half", "r20", "r80", "r_petro", "ellipticity_sm", "shape_asymmetry",
            "multimode", "intensity", "deviation", "gini_m20_merger", "sn_per_pixel", "flag", "flag_sersic"]

_STATMORPH_MAP = {"gini": "gini", "m20": "m20", "concentration": "concentration", "asymmetry": "asymmetry",
                  "smoothness": "smoothness", "sersic_n": "sersic_n", "sersic_rhalf": "sersic_rhalf",
                  "sersic_ellip": "sersic_ellip", "r_half": "rhalf_circ", "r20": "r20", "r80": "r80",
                  "r_petro": "rpetro_circ", "ellipticity_sm": "ellipticity_asymmetry",
                  "shape_asymmetry": "shape_asymmetry", "multimode": "multimode", "intensity": "intensity",
                  "deviation": "deviation", "gini_m20_merger": "gini_m20_merger", "sn_per_pixel": "sn_per_pixel",
                  "flag": "flag", "flag_sersic": "flag_sersic"}


def morph_image(cube):
    """Sum of the PSF-matched bands -> one image with the best signal-to-noise."""
    return np.asarray(cube, np.float32).sum(0)


def richardson_lucy(img, psf, iters=10):
    obs = np.clip(img, 0, None) + 1e-6
    est = np.full_like(obs, obs.mean())
    psf_m = psf[::-1, ::-1]
    for _ in range(iters):
        est *= fftconvolve(obs / (fftconvolve(est, psf, mode="same") + 1e-12), psf_m, mode="same")
    return est


def _numpy_fallback(img, target):
    pix = np.sort(np.abs(img[target]))
    n, tot = pix.size, pix.sum()
    gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * pix) / (tot * (n - 1))) if n > 1 and tot > 0 else np.nan
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2).ravel()
    o = np.argsort(r)
    cog = np.cumsum(np.clip(np.where(target, img, 0).ravel()[o], 0, None))
    out = {k: np.nan for k in FEATURES}
    if cog[-1] > 0:
        cog = cog / cog[-1]
        r20, r50, r80 = (r[o][np.searchsorted(cog, q)] for q in (0.2, 0.5, 0.8))
        out.update(r20=r20, r_half=r50, r80=r80, concentration=5 * np.log10(r80 / r20) if r20 > 0 else np.nan)
    rot = np.where(target, img, 0)[::-1, ::-1]
    den = np.abs(np.where(target, img, 0)).sum()
    out.update(gini=gini, asymmetry=float(np.abs(np.where(target, img, 0) - rot).sum() / (2 * den)) if den > 0 else np.nan,
               flag=4, flag_sersic=1)
    return out


def clean_result(res, n_bounds=(0.05, 8.0), rhalf_max=64.0):
    """-99 (statmorph's 'could not measure') -> NaN; a Sersic fit stuck at a limit -> flag_sersic = 1."""
    out = {k: (np.nan if (k not in ("flag", "flag_sersic") and v is not None and np.isfinite(v) and v <= -98.9)
               else v) for k, v in res.items()}
    n, rh = out.get("sersic_n", np.nan), out.get("sersic_rhalf", np.nan)
    stuck = (np.isfinite(n) and (n <= n_bounds[0] or n >= n_bounds[1])) or (np.isfinite(rh) and rh > rhalf_max)
    if stuck or not np.isfinite(n):
        out["flag_sersic"] = max(1.0, float(np.nan_to_num(out.get("flag_sersic", 1.0), nan=1.0)))
    return out


def measure_one(cube, seg, gain, psf=None, mode="psf_fit", rl_iters=10, sersic_maxiter=500,
                n_bounds=(0.05, 8.0), rhalf_max=64.0):
    return clean_result(_measure_one(cube, seg, gain, psf, mode, rl_iters, sersic_maxiter), n_bounds, rhalf_max)


def _measure_one(cube, seg, gain, psf=None, mode="psf_fit", rl_iters=10, sersic_maxiter=500):
    img = morph_image(cube)
    target = seg == 1                                   # cosmic-ray pixels (3) are NOT part of the target
    if target.sum() < 5:
        return {**{k: np.nan for k in FEATURES}, "flag": 4, "flag_sersic": 1}
    mask = (seg == 2) | (seg == 3) | ((img == 0) & ~target)
    use_psf = psf
    if mode == "deconvolve" and psf is not None:
        dil = np.zeros_like(target)
        from scipy.ndimage import binary_dilation
        dil = binary_dilation(target, iterations=3)
        img = np.where(dil, richardson_lucy(np.where(dil, img, 0), psf, rl_iters), img)
        use_psf = None
    if not _HAVE_STATMORPH:
        return _numpy_fallback(img, target)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = statmorph.source_morphology(img, target.astype(int), mask=mask, gain=gain, psf=use_psf,
                                            sersic_maxiter=sersic_maxiter)[0]
        return {k: float(getattr(m, v)) if getattr(m, v, None) is not None else np.nan
                for k, v in _STATMORPH_MAP.items()}
    except Exception:
        return _numpy_fallback(img, target)


def measure_batch(cut_path, seg_path, indices, gains, psfs, mode, rl_iters, sersic_maxiter,
                  n_bounds=(0.05, 8.0), rhalf_max=64.0):
    """Worker for joblib: opens the memory-mapped arrays itself (nothing big is pickled)."""
    try:
        from astropy import log as _alog
        _alog.setLevel("ERROR")                    # statmorph reports through astropy's logger
    except Exception:
        pass
    warnings.simplefilter("ignore")
    cut = np.load(cut_path, mmap_mode="r")
    seg = np.load(seg_path, mmap_mode="r")
    out = []
    for i, g, p in zip(indices, gains, psfs):
        out.append(measure_one(cut[i].astype(np.float32), np.asarray(seg[i]), g, p, mode, rl_iters, sersic_maxiter,
                               n_bounds, rhalf_max))
    return out


def run_parallel(ctx, cut_path, seg_path, indices, gains, psfs, mode="psf_fit"):
    from joblib import Parallel, delayed
    cfg = ctx.cfg["morphology"]
    n_jobs = ctx.n_jobs
    chunks = np.array_split(np.arange(len(indices)), max(1, min(len(indices), n_jobs * 4)))
    jobs = [delayed(measure_batch)(str(cut_path), str(seg_path), [indices[j] for j in ch], [gains[j] for j in ch],
                                   [psfs[j] for j in ch], mode, cfg["rl_iterations"], cfg["sersic_maxiter"],
                                   tuple(cfg.get("sersic_n_bounds", (0.05, 8.0))),
                                   float(cfg.get("sersic_rhalf_max_px", 64.0)))
            for ch in chunks if len(ch)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
    return [r for part in res for r in part]
