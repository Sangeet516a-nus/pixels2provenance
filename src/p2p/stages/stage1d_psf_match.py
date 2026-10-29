"""Stage 1d - Cross-filter PSF matching (no learned model).

Purpose : give every band of a cutout the same resolution, so colour differences in the PSF are not read as
          galaxy structure.
Method  : 1. PSF stars per pointing: S/N >= star_min_snr, round (ellipticity < star_max_ellipticity), not
             saturated, not a cosmic-ray candidate, seen in a second band, Gaussian-fit FWHM within 15 % of the
             stellar locus (median of the Gaia stars, else the peak of the size distribution near the expected
             width), core free of neighbours (star_isolation_px).
          2. Per band, stars with S/N >= star_min_snr_per_band are recentred on the detection-band core and
             median-stacked; the band FWHM = median of the stars' 2-D Gaussian fits. With fewer than min_stars
             the band BORROWS the stack of the pointing with most stars in that band (same camera, filter, pixel
             scale); a Gaussian of the instrument width only if no pointing has stars. No stars anywhere -> stop
             (require_stars).
          3. The broadest band is the target; every sharper band is convolved with a Gaussian kernel,
             sigma_K^2 = sigma_target^2 - sigma_band^2 (or photutils' Fourier-ratio kernel: kernel: photutils).
Output  : cutouts.npy (PSF-matched, used by all later stages), psf_match.json, psf_<pointing>_<band>.npy,
          placeholder_psf_<pointing>.npy (Gaussian of the target FWHM for Stage 1e; the empirical PSF is Stage 5a)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve

from ..context import Ctx, load_json, save_json
from ..utils import as_bool, fwhm_fit, gaussian_psf

def crop_center(img, n):
    c = img.shape[0] // 2
    h = n // 2
    return img[c - h:c + h + 1, c - h:c + h + 1]


def core_centroid(stamp, half=3):
    """Flux centroid of the central (2*half+1)^2 pixels only (neighbours cannot pull it)."""
    s = np.clip(np.asarray(stamp, float), 0, None)
    c = s.shape[0] // 2
    box = s[c - half:c + half + 1, c - half:c + half + 1]
    if box.sum() <= 0:
        return float(c), float(c)
    cy, cx = ndimage.center_of_mass(box)
    return float(cy + c - half), float(cx + c - half)


def stack_stars(stamps, centers=None, masks=None):
    """Recentre each star (cubic sub-pixel shift) on centers (detection-band cores, so every band is shifted
    identically), normalise, median; masks (True = other source) are left out."""
    n = stamps[0].shape[0]
    c = (n - 1) / 2.0
    out = []
    for k, s in enumerate(stamps):
        s = np.asarray(s, float)
        cy, cx = centers[k] if centers is not None else core_centroid(s)
        bad = np.asarray(masks[k], bool) if masks is not None else np.zeros_like(s, bool)
        s = ndimage.shift(np.where(bad, 0.0, s), (c - cy, c - cx), order=3, mode="constant")
        bad = ndimage.shift(bad.astype(float), (c - cy, c - cx), order=0, mode="constant", cval=1.0) > 0.5
        tot = s[~bad].sum()
        if tot > 0:
            out.append(np.where(bad, np.nan, s / tot))
    psf = np.nan_to_num(np.clip(np.nanmedian(out, axis=0), 0, None))
    return psf / psf.sum()


def star_masks(segs, idx, n):
    """Pixels of other sources (2) in each star stamp. Cosmic-ray pixels (3) are not masked: Stage 1c already
    replaced them in the band they hit."""
    return [crop_center(np.asarray(segs[i]), n) == 2 for i in idx]


def flat_top(stamp, tol=0.02):
    top = np.sort(np.asarray(stamp, float).ravel())[-5:]
    return (top[-1] - top[0]) / max(abs(top[-1]), 1e-12) < tol


def matching_kernel(psf_sharp, psf_broad, beta, method="gaussian", fwhm_sharp=None, fwhm_broad=None):
    """gaussian (default): Gaussian K with sigma_K^2 = sigma_broad^2 - sigma_sharp^2 (robust for noisy stacks);
    photutils: Fourier-ratio kernel with a TopHat window (needs clean PSFs)."""
    if method == "gaussian":
        fs = fwhm_sharp if fwhm_sharp is not None else fwhm_fit(psf_sharp)
        fb = fwhm_broad if fwhm_broad is not None else fwhm_fit(psf_broad)
        fk = np.sqrt(max(fb ** 2 - fs ** 2, 0.0))
        if fk < 0.3:
            k = np.zeros_like(psf_sharp)
            k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
            return k
        return gaussian_psf(fk, psf_sharp.shape[0])
    from photutils.psf.matching import TopHatWindow, create_matching_kernel
    return create_matching_kernel(psf_sharp, psf_broad, window=TopHatWindow(beta))


def convolve_stamp(stamp, K):
    pad = K.shape[0] // 2
    padded = np.pad(stamp, pad, mode="reflect")
    return fftconvolve(padded, K, mode="same")[pad:-pad, pad:-pad]


def gaia_flags(ctx, src, gaia):
    """True for sources within gaia_radius_arcsec of a Gaia DR3 star."""
    if gaia is None or gaia.empty:
        return np.zeros(len(src), bool)
    from ..utils import sky_crossmatch
    return sky_crossmatch(src.ra, src.dec, gaia.ra, gaia.dec, ctx.cfg["semantic_audit"]["gaia_radius_arcsec"])[2]


def load_gaia_once(ctx):
    """Gaia DR3 stars of the field (data/catalogs), read once for all pointings; None if unavailable."""
    from ..catalogs import load_gaia
    try:
        return load_gaia(ctx, list(load_json(ctx.require("1a", "manifest.json")).values()))
    except Exception as e:
        ctx.log.info(f"[Stage 1d] WARNING: Gaia catalogue not usable ({e!r}); stellar locus from the size peak.")
        return None


def select_stars(src: pd.DataFrame, segs, raw, bands, present, cfg, expected_px, gaia=None):
    """Bright, round, isolated, compact sources detected in every band that COVERS them and in at least
    one band besides the detection band (a cosmic ray is as compact as a star but appears in one band only)."""
    bright = src[(src.snr >= cfg["star_min_snr"]) & (~src.edge_flag.astype(bool))
                 & (src.ellipticity < cfg.get("star_max_ellipticity", 0.3))]
    if "cr_candidate" in bright:                # never a cosmic ray (no second band confirms it)
        bright = bright[~as_bool(bright.cr_candidate)]
    if "saturated" in bright:
        bright = bright[~as_bool(bright.saturated)]
    det = src.detection_band.iloc[0] if len(src) else None
    others = [b for b in present if b != det]
    seen_in_other = np.zeros(len(bright), bool)
    for b in others:
        cov = bright[f"covered_{b}"].astype(bool) if f"covered_{b}" in bright else pd.Series(True, index=bright.index)
        ok_b = bright[f"flux_{b}"] > 0.2 * bright[f"flux_{det}"]
        keep = ~cov | ok_b                      # detected in every band that COVERS it ...
        seen_in_other = seen_in_other[keep.to_numpy()] | ok_b[keep].to_numpy()
        bright = bright[keep]
    if others:
        bright = bright[seen_in_other]          # ... and in at least one other band (cosmic rays are single-band)
    if len(bright) == 0:
        return bright, np.nan
    # size from a 2-D Gaussian fit of the detection-band stamp (the segment-moment FWHM depends on the isophote)
    det_i = bands.index(det)
    size = []
    for i in bright.cutout_index:
        img = np.where(np.asarray(segs[i]) == 2, 0.0, np.asarray(raw[i, det_i], np.float32))
        size.append(fwhm_fit(crop_center(img, cfg["stamp"])))
    bright = bright.assign(fit_fwhm_px=size)
    # stellar locus: median size of >= 2 Gaia stars, else the peak of the size distribution near the expected
    # PSF width (cosmic-ray coincidences are sharper, compact galaxies broader)
    g = bright.index.isin(src.index[gaia]) if gaia is not None else np.zeros(len(bright), bool)
    if g.sum() >= 2:
        locus = float(np.median(bright.fit_fwhm_px[g]))
    else:
        lo, hi = 0.85 * expected_px, 1.5 * expected_px
        inwin = bright.fit_fwhm_px.between(lo, hi)
        if inwin.sum() < 3:
            return bright.iloc[:0], np.nan
        h, e = np.histogram(bright.fit_fwhm_px[inwin], bins=np.arange(lo, hi + 0.1, 0.1))
        h = np.convolve(h, np.ones(3) / 3, mode="same")
        locus = float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))
    cand = bright[np.abs(bright.fit_fwhm_px - locus) <= 0.15 * locus]
    S = segs.shape[1]
    yy, xx = np.mgrid[:S, :S]
    # only the core must be free of other segments; neighbours further out are masked in the stack
    near = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2) <= cfg.get("star_isolation_px", 5)
    iso = [not np.any(segs[i][near] == 2) for i in cand.cutout_index]
    cand = cand[np.asarray(iso, bool)]
    return cand.nlargest(cfg["max_stars"], "snr"), float(locus)


def _fallback_fw(cfg, p):
    fb = cfg["fallback_fwhm_arcsec"]
    return fb.get(p["instrument"], fb.get(p["instrument"].split("_")[0], fb["default"]))


def measure_pointing(ctx, cfg, pid, p, rows, bands, raw, segs, gaia):
    """PSF stars and per-band stacked PSFs of one pointing (bands with too few stars are left out)."""
    expected = _fallback_fw(cfg, p) / float(rows.pixscale_out.iloc[0])
    stars, locus = select_stars(rows, segs, raw, bands, list(p["bands"]), cfg, expected, gaia_flags(ctx, rows, gaia))
    det_i = bands.index(p["detection_band"])
    stars = stars[np.array([not flat_top(raw[i, det_i]) for i in stars.cutout_index], dtype=bool)]
    if len(stars):                              # Gaussian-fit size of the PSF stars (same estimator as the PSFs)
        locus = float(np.median(stars.fit_fwhm_px))
    psfs, method, star_fwhm, n_per_band = {}, {}, {}, {}
    for bi, b in enumerate(bands):
        if b not in p["bands"]:
            continue
        in_b = stars[stars[f"covered_{b}"].astype(bool)] if f"covered_{b}" in stars else stars
        if f"segsnr_{b}" in in_b:              # a red star is too faint in a blue band to shape its PSF
            in_b = in_b[in_b[f"segsnr_{b}"].fillna(0) >= cfg.get("star_min_snr_per_band", 30)]
        n_per_band[b] = int(len(in_b))
        if len(in_b) >= cfg["min_stars"]:
            idx = in_b.cutout_index.tolist()
            cen = [core_centroid(crop_center(raw[i, det_i].astype(np.float32), cfg["stamp"])) for i in idx]
            masks = star_masks(segs, idx, cfg["stamp"])
            crops = [crop_center(raw[i, bi].astype(np.float32), cfg["stamp"]) for i in idx]
            psfs[b] = stack_stars(crops, cen, masks)
            star_fwhm[b] = float(np.median([fwhm_fit(np.where(m, 0.0, c)) for c, m in zip(crops, masks)]))
            method[b] = f"stacked {len(in_b)} stars"
    if psfs:                                    # a stack far broader than the other bands = too few clean stars
        fw = {b: star_fwhm.get(b, fwhm_fit(v)) for b, v in psfs.items()}
        med = float(np.median(list(fw.values())))
        for b in [b for b, f in fw.items() if f > 1.6 * med and len(fw) > 1]:
            ctx.log.info(f"[Stage 1d] {pid}: WARNING - the {b} PSF stack ({fw[b]:.2f} px) is > 1.6 x the median of the "
                         f"bands ({med:.2f} px): too few clean stars in {b}; a borrowed / fallback PSF is used for it.")
            psfs.pop(b)
            method.pop(b)
            star_fwhm.pop(b, None)
    return {"stars": stars, "locus": locus, "psfs": psfs, "method": method, "star_fwhm": star_fwhm,
            "n_per_band": n_per_band, "scale": float(rows.pixscale_out.iloc[0])}


def run(ctx: Ctx):
    cfg = ctx.cfg["psf_match"]
    out = ctx.stage_dir("1d")
    info = load_json(ctx.require("1a", "pointings.json"))
    bands, pts = info["bands"], info["pointings"]
    src = pd.read_csv(ctx.require("1c", "sources.csv"))
    raw = np.load(ctx.require("1c", "cutouts_raw.npy"), mmap_mode="r")
    segs = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    matched = np.lib.format.open_memmap(out / "cutouts.npy", mode="w+", dtype=np.float16, shape=raw.shape)

    # pass 1: measure every pointing
    gaia = load_gaia_once(ctx)
    meas = {}
    for pid, p in pts.items():
        rows = src[src.pointing == pid]
        if not rows.empty:
            meas[pid] = measure_pointing(ctx, cfg, pid, p, rows, bands, raw, segs, gaia)

    # library of measured PSFs per (instrument, pixel scale, band): the pointing with most stars in that band
    library = {}
    for pid, m in meas.items():
        for b, psf in m["psfs"].items():
            key = (pts[pid]["instrument"], round(m["scale"], 4), b)
            n = m["n_per_band"][b]
            if key not in library or n > library[key][2]:
                library[key] = (psf, m["star_fwhm"].get(b, fwhm_fit(psf)), n, pid)
    if not library and cfg.get("require_stars", True):
        raise RuntimeError(
            f"[Stage 1d] no PSF star passed the cuts in ANY pointing (SNR >= {cfg['star_min_snr']}, round, "
            "isolated, unsaturated, confirmed in a second band). The PSFs would be the configured Gaussian, not a "
            "measurement. Check the Stage 1c registration / cosmic-ray log lines; to continue with the Gaussian "
            "fallback anyway set psf_match.require_stars: false (the catalogue then stores no measured matched FWHM).")

    # pass 2: fill missing bands (borrow, else Gaussian), match, write cutouts
    report = {}
    for pid, m in meas.items():
        p = pts[pid]
        rows = src[src.pointing == pid]
        stars, locus, scale = m["stars"], m["locus"], m["scale"]
        psfs, method, star_fwhm = dict(m["psfs"]), dict(m["method"]), dict(m["star_fwhm"])
        for b in bands:
            if b not in p["bands"] or b in psfs:
                continue
            key = (p["instrument"], round(scale, 4), b)
            if key in library:
                psf, fw_px, n, src_pid = library[key]
                psfs[b], star_fwhm[b] = psf, fw_px
                method[b] = f"borrowed from {src_pid} ({n} stars; only {m['n_per_band'].get(b, 0)} here)"
            else:
                fw = _fallback_fw(cfg, p)
                psfs[b] = gaussian_psf(fw / scale, cfg["stamp"])
                method[b] = f"Gaussian fallback ({fw}\" FWHM; only {m['n_per_band'].get(b, 0)} stars cover {b})"
        if any(v.startswith("borrowed") for v in method.values()):
            ctx.log.info(f"[Stage 1d] {pid}: {len(stars)} PSF stars here - PSFs borrowed from a pointing with the same "
                         f"camera, filter and pixel scale: { {b: v for b, v in method.items() if v.startswith('borrowed')} }")
        for b, v in psfs.items():
            np.save(out / f"psf_{pid}_{b}.npy", v)
        fwhm = {b: star_fwhm.get(b, fwhm_fit(v)) for b, v in psfs.items()}
        target = max(fwhm, key=fwhm.get)
        kernels = {b: matching_kernel(psfs[b], psfs[target], cfg["window"], cfg.get("kernel", "gaussian"),
                                      fwhm[b], fwhm[target]) for b in psfs if b != target}
        # FWHM after matching: Gaussian kernel -> widths add in quadrature (no kernel if f_K < 0.3 px);
        # photutils kernel -> fit of the convolved stack
        def after_matching(b, K):
            if cfg.get("kernel", "gaussian") != "gaussian":
                return fwhm_fit(fftconvolve(psfs[b], K, mode="same"))
            fk = np.sqrt(max(fwhm[target] ** 2 - fwhm[b] ** 2, 0.0))
            return float(np.hypot(fwhm[b], fk)) if fk >= 0.3 else fwhm[b]
        check = {b: round(after_matching(b, K), 3) for b, K in kernels.items()}
        for i in rows.cutout_index:
            cube = raw[i].astype(np.float32)
            for bi, b in enumerate(bands):
                if b in kernels:
                    cube[bi] = convolve_stamp(cube[bi], kernels[b])
            matched[i] = cube.astype(np.float16)
        np.save(out / f"placeholder_psf_{pid}.npy", gaussian_psf(fwhm[target], cfg["stamp"]))
        if not np.isfinite(locus):          # no bright compact sources: use the PSF width
            locus = fwhm[target]
        measured = method[target].startswith(("stacked", "borrowed"))
        if not measured:
            ctx.log.info(f"[Stage 1d] {pid}: WARNING - the target band {target} uses the Gaussian fallback; "
                         "no measured matched FWHM is stored.")
        report[pid] = {"n_stars": int(len(stars)), "star_ids": stars.object_id.tolist(), "method": method,
                       "target_psf_measured": bool(measured),
                       "target_psf_borrowed": method[target].startswith("borrowed"),
                       "matched_fwhm_arcsec": round(fwhm[target] * scale, 4) if measured else None,
                       "stellar_locus_fwhm_px": round(float(locus), 3),
                       "fwhm_px": {b: round(v, 3) for b, v in fwhm.items()}, "target_band": target,
                       "matched_fwhm_px": check, "pixscale_arcsec": scale}
        ctx.log.info(f"[Stage 1d] {pid}: stars={len(stars)} FWHM(px)={report[pid]['fwhm_px']} -> target {target}; "
                     f"after matching {check}")
    matched.flush()
    save_json(report, out / "psf_match.json")
    _plot(report, out)
    return {"pointings": len(report)}


def _plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for pid in report:
        files = sorted(out.glob(f"psf_{pid}_*.npy"))
        if not files:
            continue
        fig, axes = plt.subplots(1, len(files), figsize=(3 * len(files), 3))
        for ax, f in zip(np.atleast_1d(axes), files):
            ax.imshow(np.log10(np.load(f) + 1e-5), origin="lower", cmap="magma")
            b = f.stem.split("_")[-1]
            ax.set_title(f"{b}: FWHM {report[pid]['fwhm_px'][b]:.2f} px", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Stage 1d {pid}: per-band PSFs (log)")
        fig.tight_layout()
        fig.savefig(out / f"{pid}_psfs.png", dpi=100)
        plt.close(fig)
