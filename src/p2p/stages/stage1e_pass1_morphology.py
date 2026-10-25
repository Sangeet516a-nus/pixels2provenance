"""Stage 1e - Pass-1 photometry and morphology.

Purpose : a provisional table of physical parameters for every source.
Method  : statmorph on the sum of the PSF-matched bands, with the target
          segment from Stage 1c and every neighbouring segment masked. The
          Sersic fit is convolved with a generic PLACEHOLDER PSF (Gaussian with
          the Stage-1d target FWHM); Stage 5b re-runs the galaxy track with the
          empirical PSF. Sources below morphology.min_snr are not measured
          (NaN, flag 4). Gain = detection-band EXPTIME (images are in e-/s).
          fwhm_over_psf = source FWHM / PSF FWHM (1 = point-like), both from a
          2-D Gaussian fit to the detection-band cutout (neighbours removed,
          cosmic-ray pixels already replaced by Stage 1c) and to the Stage-1d PSF of that band. A fit
          does not shrink with S/N the way a segment-moment FWHM does (faint
          galaxies only show their core above the threshold and looked
          point-like or even sharper than the PSF).
Output  : features_pass1.csv  (psf_pass = 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_json
from ..morphology import FEATURES, _HAVE_STATMORPH, run_parallel
from ..utils import fwhm_fit
from .stage1d_psf_match import crop_center


def fit_fwhm_ratio(ctx, src):
    """Gaussian-fit FWHM of every source / of its pointing's detection-band PSF (Stage 1d)."""
    raw = np.load(ctx.require("1c", "cutouts_raw.npy"), mmap_mode="r")
    seg = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    bands = load_json(ctx.require("1a", "pointings.json"))["bands"]
    stamp = ctx.cfg["psf_match"]["stamp"]
    locus = {}
    rep = load_json(ctx.require("1d", "psf_match.json"))
    for pid, det in src.groupby("pointing").detection_band.first().items():
        locus[pid] = rep[pid]["fwhm_px"].get(det) or fwhm_fit(np.load(ctx.require("1d", f"psf_{pid}_{det}.npy")))
    out = np.full(len(src), np.nan)
    for k, (i, p, det) in enumerate(zip(src.cutout_index, src.pointing, src.detection_band)):
        img = np.asarray(raw[int(i), bands.index(det)], np.float32)
        img = np.where(np.asarray(seg[int(i)]) == 2, 0.0, img)    # neighbours (cosmic-ray pixels: already replaced)
        out[k] = fwhm_fit(crop_center(img, stamp)) / locus[p]
    return out, locus


def run(ctx: Ctx):
    out = ctx.stage_dir("1e")
    src = pd.read_csv(ctx.require("1c", "sources.csv"))
    cut_path = ctx.require("1d", "cutouts.npy")
    seg_path = ctx.require("1c", "segstamps.npy")
    min_snr = ctx.cfg["morphology"]["min_snr"]
    todo = src[src.snr >= min_snr]
    psf = {pid: np.load(ctx.require("1d", f"placeholder_psf_{pid}.npy")) for pid in todo.pointing.unique()}
    ctx.log.info(f"[Stage 1e] measuring {len(todo)} of {len(src)} sources (SNR >= {min_snr}) with "
                 f"{'statmorph' if _HAVE_STATMORPH else 'NumPy fallback'} on {ctx.n_jobs} worker(s) ...")
    res = run_parallel(ctx, cut_path, seg_path, todo.cutout_index.tolist(), todo.exptime.tolist(),
                       [psf[p] for p in todo.pointing], mode="psf_fit")
    feats = pd.DataFrame(res, index=todo.index)
    table = pd.DataFrame(index=src.index, columns=FEATURES, dtype=float)
    table.loc[feats.index, FEATURES] = feats[FEATURES].to_numpy()
    table["flag"] = table["flag"].fillna(4)
    table.insert(0, "object_id", src.object_id)
    table["ellipticity"] = src["ellipticity"]           # from the Stage-1c segment moments
    # size relative to the PSF, measured with the SAME estimator (2-D Gaussian fit) on source and PSF
    ratio, locus = fit_fwhm_ratio(ctx, src)
    table["fwhm_over_psf"] = np.round(ratio, 4)
    ctx.log.info(f"[Stage 1e] PSF FWHM (Gaussian fit, detection band) per pointing: "
                 f"{ {p: round(v, 3) for p, v in locus.items()} } px | fwhm_over_psf median {np.nanmedian(ratio):.2f}, "
                 f"share point-like (0.8-1.2) {np.mean(np.abs(ratio - 1) <= 0.2):.1%}")
    table["psf_pass"] = 1
    table["morph_engine"] = "statmorph" if _HAVE_STATMORPH else "numpy_fallback"
    table.to_csv(out / "features_pass1.csv", index=False)
    good = table[table.flag <= 1]
    ctx.log.info(f"[Stage 1e] flags: {table.flag.value_counts().sort_index().to_dict()} | "
                 f"median Gini {good.gini.median():.3f}, C {good.concentration.median():.2f}, "
                 f"A {good.asymmetry.median():.3f}, n {good.sersic_n.median():.2f}")
    _plot(table, out)
    return {"measured": int(len(todo)), "good_flag": int(len(good))}


def _plot(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = t[t.flag <= 1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(g.m20, g.gini, s=4, alpha=0.5)
    xs = np.array([-3.0, 0.0])
    ax[0].plot(xs, -0.14 * xs + 0.33, "k--", lw=1, label="merger line (Lotz+08)")
    ax[0].plot(xs, 0.14 * xs + 0.80, "k:", lw=1, label="E/S0 vs Sb line")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("M20")
    ax[0].set_ylabel("Gini")
    ax[0].legend(fontsize=7)
    ax[1].scatter(g.asymmetry, g.concentration, s=4, alpha=0.5)
    ax[1].set_xlabel("asymmetry A")
    ax[1].set_ylabel("concentration C")
    fig.suptitle("Stage 1e: Pass-1 morphology (flag <= 1)")
    fig.tight_layout()
    fig.savefig(out / "pass1_morphology.png", dpi=110)
    plt.close(fig)
