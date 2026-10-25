"""Stage 5a - Empirical PSF construction (photutils EPSFBuilder).

Purpose : a localised, high-quality PSF built from the pipeline's own stars.
Method  : the stars routed by Stage 3b are cut to 31x31 stamps from the image
          that is actually measured (the sum of the PSF-matched bands) and
          stacked with photutils.psf.EPSFBuilder (oversampling 2, sub-pixel
          recentring, sigma-clipped). Pointings with too few stars pool the
          stars of all pointings with the same instrument; if that is still too
          few, the Pass-1 placeholder PSF is kept (and the report says so).
Output  : epsf_<pointing>.npy (+ .fits), epsf_report.json, epsf.png
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from astropy.io import fits

from ..context import Ctx, load_json, save_json
from ..features import load_cutouts
from ..morphology import morph_image
from ..utils import as_bool, fwhm_fit
from .stage1d_psf_match import core_centroid, crop_center, stack_stars, star_masks

try:
    from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars
    _HAVE_EPSF = True
except Exception:
    _HAVE_EPSF = False


def build_epsf(stamps, masks=None, oversampling=2, maxiters=5):
    """masks: True = neighbour / cosmic-ray pixel (weight 0). Stars are centred on their core centroid."""
    masks = masks if masks is not None else [np.zeros_like(s, bool) for s in stamps]
    if _HAVE_EPSF:
        try:
            stars = EPSFStars([EPSFStar(np.where(m, 0.0, s).astype(float), weights=(~m).astype(float),
                                        cutout_center=core_centroid(np.where(m, 0.0, s))[::-1])
                               for s, m in zip(stamps, masks)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, progress_bar=False)(stars)
            epsf = res.epsf if hasattr(res, "epsf") else res[0]
            n = stamps[0].shape[0]
            yy, xx = np.mgrid[:n, :n]
            psf = np.clip(epsf.evaluate(xx, yy, flux=1.0, x_0=(n - 1) / 2, y_0=(n - 1) / 2), 0, None)
            return psf / psf.sum(), "photutils.EPSFBuilder"
        except Exception as e:
            print(f"[Stage 5a] EPSFBuilder failed ({e}); median stack used")
    return stack_stars(stamps, masks=masks), "median stack"


def run(ctx: Ctx, size=31, min_stars=5):
    out = ctx.stage_dir("5a")
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    routing = pd.read_csv(ctx.require("3b", "star_routing.csv"))
    src = pd.read_csv(ctx.require("1c", "sources.csv"))[["object_id", "cutout_index", "pointing"]]
    routed = routing[as_bool(routing.used_in_epsf)].merge(src, on=["object_id", "pointing"])
    cut = load_cutouts(ctx)
    segs = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    match = load_json(ctx.require("1d", "psf_match.json"))
    stamps = {r.object_id: crop_center(morph_image(cut[r.cutout_index]), size) for r in routed.itertuples()}
    report = {}
    for pid, p in pts.items():
        own = routed[routed.pointing == pid].object_id.tolist()
        use, how = own, "own stars"
        if len(own) < min_stars:
            same = [q for q, v in pts.items() if v["instrument"] == p["instrument"]]
            use = routed[routed.pointing.isin(same)].object_id.tolist()
            how = f"pooled over {len(same)} pointing(s) of {p['instrument']}"
        if len(use) >= min_stars:
            idx = routed.set_index("object_id").loc[use, "cutout_index"].tolist()
            psf, method = build_epsf([stamps[o] for o in use], star_masks(segs, idx, size))
        else:
            ph = np.load(ctx.require("1d", f"placeholder_psf_{pid}.npy"))
            psf = np.zeros((size, size))
            h = (size - ph.shape[0]) // 2
            psf[h:h + ph.shape[0], h:h + ph.shape[1]] = ph
            method, how = "Pass-1 placeholder kept", f"only {len(use)} star(s) available"
        np.save(out / f"epsf_{pid}.npy", psf.astype(np.float32))
        fits.writeto(out / f"epsf_{pid}.fits", psf.astype(np.float32), overwrite=True)
        report[pid] = {"n_stars": len(use), "source": how, "method": method, "fwhm_px": round(fwhm_fit(psf), 3)}
        # the ePSF is built on PSF-MATCHED cutouts, so it cannot be sharper than the matched PSF, and a much
        # broader one means galaxies or cosmic rays got into the stack
        m = match.get(pid, {})
        ref = m.get("fwhm_px", {}).get(m.get("target_band")) if m.get("target_psf_measured") else None
        if ref:
            ratio = report[pid]["fwhm_px"] / ref
            report[pid]["fwhm_over_matched"] = round(ratio, 3)
            if ratio > 1.5 or ratio < 0.85:
                ctx.log.info(f"[Stage 5a] {pid}: WARNING - ePSF FWHM {report[pid]['fwhm_px']} px is {ratio:.2f} x the "
                             f"Stage-1d matched FWHM {ref} px (expected ~1): check the routed stars "
                             "(outputs/stage3b_star_routing/star_routing.csv).")
        ctx.log.info(f"[Stage 5a] {pid}: {method} from {len(use)} stars ({how}); FWHM {report[pid]['fwhm_px']} px")
    save_json(report, out / "epsf_report.json")
    _plot(report, out)
    return {"pointings": len(report)}


def _plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pids = list(report)
    fig, axes = plt.subplots(1, len(pids), figsize=(2.6 * len(pids), 2.8))
    for ax, pid in zip(np.atleast_1d(axes), pids):
        ax.imshow(np.log10(np.load(out / f"epsf_{pid}.npy") + 1e-5), origin="lower", cmap="magma")
        ax.set_title(f"{pid}: {report[pid]['n_stars']} stars\nFWHM {report[pid]['fwhm_px']} px", fontsize=8)
        ax.axis("off")
    fig.suptitle("Stage 5a: empirical PSFs (log scale)")
    fig.tight_layout()
    fig.savefig(out / "epsf.png", dpi=100)
    plt.close(fig)
