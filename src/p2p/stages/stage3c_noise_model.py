"""Stage 3c - Noise: false-positive modelling.

Purpose : characterise spurious detections - the Noise population has no
          external catalogue, so this model is also its validation.
Method  : two kinds of spurious detection are modelled separately:
          (1) sky-noise peaks - the NEGATIVE-IMAGE test on each detection frame.
              Sky noise is symmetric, so the number of 'sources' found on -SCI
              at nu x sigma estimates how many detections on +SCI at the same
              threshold are spurious. nu is each source's PEAK significance in
              the smoothed detection image (the quantity the detection threshold
              is applied to), not its total SNR. The grid of thresholds extends
              to the brightest source, so bright sources are not clamped to the
              last grid value. The FPR cannot be resolved below 1 / n_pos(nu)
              (one negative detection); that floor is reported, and an FPR of 0
              means 'below the floor'.
          (2) cosmic rays - positive-only, so the negative-image test cannot see
              them (single exposures are full of them, at HIGH S/N). Their term
              is p_cr = sigmoid((confirm_snr - best other-band S/N) / 0.5) from
              the Stage-1c two-band test (0 when no other band covers the source).
          (3) a profile sharper than the PSF: p_sub = sigmoid((sub_psf_factor - FWHM/locus) / 0.05).
          noise_score = 1 - (1 - FPR(nu)) (1 - p_sub) (1 - p_cr).
Output  : noise_model.json (FPR curve per pointing, operating threshold, resolution floor),
          noise_scores.csv, fpr_curve.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..compat import detect, n_labels
from ..context import Ctx, load_json, load_objects, save_json
from ..detection import smooth, subtract_background
from ..fitsio import load_image, valid_mask


def threshold_grid(nu_max, lo=1.5, fine_to=6.0, step=0.25, n_log=20):
    """Fine linear steps to 6 sigma, then logarithmic steps up to the brightest source."""
    fine = np.arange(lo, fine_to + 1e-9, step)
    if not np.isfinite(nu_max) or nu_max <= fine_to:
        return fine
    return np.unique(np.concatenate([fine, np.geomspace(fine_to, nu_max * 1.05, n_log)]))


def run(ctx: Ctx, target_fpr=0.01):
    out = ctx.stage_dir("3c")
    dcfg = ctx.cfg["detection"]
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    df = load_objects(ctx, ["sources", "baseline", "segregation"])
    nu = np.full(len(df), np.nan)
    curves = {}
    for pid, p in pts.items():
        sci, wht, _ = load_image(p["bands"][p["detection_band"]])
        val = valid_mask(sci, wht)
        sub, rms = subtract_background(sci, val, dcfg["background_box"], dcfg["background_filter"])
        conv = smooth(sub, dcfg["kernel_fwhm_px"])                 # same filter as Stage 1c, done once
        base = rms * dcfg.get("correlated_noise_factor", 1.0)
        m = (df.pointing == pid).to_numpy()
        xi = np.clip(np.round(df.x[m].to_numpy()).astype(int), 0, conv.shape[1] - 1)
        yi = np.clip(np.round(df.y[m].to_numpy()).astype(int), 0, conv.shape[0] - 1)
        nu[m] = conv[yi, xi] / base[yi, xi]                        # peak significance at the centroid
        grid = threshold_grid(np.nanmax(nu[m]) if m.any() else np.nan)
        rows = []
        for k in grid:
            n_pos = n_labels(detect(conv, k * base, dcfg["npixels"], mask=~val))
            n_neg = n_labels(detect(-conv, k * base, dcfg["npixels"], mask=~val))
            rows.append({"nsigma": float(k), "n_pos": n_pos, "n_neg": n_neg,
                         "fpr": float(min(n_neg / max(n_pos, 1), 1.0)),
                         "fpr_floor": float(1.0 / max(n_pos, 1))})
            if n_pos == 0:
                break
        cur = pd.DataFrame(rows)
        ok = cur[cur.fpr <= target_fpr]
        last_neg = cur[cur.n_neg > 0]
        floor_nu = float(last_neg.nsigma.max()) if len(last_neg) else float(cur.nsigma.min())
        curves[pid] = {"curve": cur.to_dict(orient="list"),
                       "operating_nsigma": float(ok.nsigma.iloc[0]) if len(ok) else float(cur.nsigma.iloc[-1]),
                       "fpr_at_configured_nsigma": float(np.interp(dcfg["nsigma"], cur.nsigma, cur.fpr)),
                       "last_threshold_with_a_negative_detection": floor_nu,
                       "resolution_note": "FPR = n_neg / n_pos; above the last threshold with a negative detection "
                                          "the FPR is only known to be below 1 / n_pos at that threshold"}
        ctx.log.info(f"[Stage 3c] {pid}: FPR <= {target_fpr} from {curves[pid]['operating_nsigma']:.2f} sigma; "
                     f"FPR at the detection threshold ({dcfg['nsigma']} sigma) = "
                     f"{curves[pid]['fpr_at_configured_nsigma']:.3f}; no negative detection above "
                     f"{floor_nu:.1f} sigma (FPR resolution there: 1/{int(cur.n_pos[cur.nsigma >= floor_nu].iloc[0])})")

    fpr = np.full(len(df), np.nan)
    for pid, c in curves.items():
        m = (df.pointing == pid).to_numpy()
        fpr[m] = np.interp(nu[m], c["curve"]["nsigma"], c["curve"]["fpr"], right=0.0)
    sub_psf = 1.0 / (1.0 + np.exp((df.fwhm_over_psf.to_numpy(float) - ctx.cfg["baseline"]["sub_psf_factor"]) / 0.05))
    confirm = float(dcfg.get("confirm_snr", 3.0))
    best = pd.to_numeric(df.get("confirm_snr_best", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(float)
    p_cr = np.where(np.isfinite(best), 1.0 / (1.0 + np.exp((best - confirm) / 0.5)), 0.0)
    score = 1 - (1 - np.nan_to_num(fpr)) * (1 - np.nan_to_num(sub_psf)) * (1 - p_cr)
    res = pd.DataFrame({"object_id": df.object_id, "peak_significance": np.round(nu, 3),
                        "fpr_at_threshold": np.round(fpr, 5), "p_cosmic_ray": np.round(p_cr, 4),
                        "noise_score": np.round(score, 4)})
    res.to_csv(out / "noise_scores.csv", index=False)
    validation = res.groupby(df.cluster_semantic.to_numpy())["noise_score"].mean().round(3).to_dict()
    snr_by_track = df.groupby("cluster_semantic").snr.median().round(1).to_dict()
    save_json({"target_fpr": target_fpr, "pointings": curves, "validation_mean_noise_score_by_track": validation,
               "median_snr_by_track": snr_by_track,
               "terms": "noise_score = 1 - (1 - FPR(peak significance)) (1 - p_sub_psf) (1 - p_cosmic_ray)"},
              out / "noise_model.json")
    _plot(curves, out)
    ctx.log.info(f"[Stage 3c] mean noise_score by track (validation): {validation} | median SNR by track "
                 f"{snr_by_track} (cosmic rays are high-S/N: the p_cosmic_ray term, not the FPR, scores them)")
    return {"validation": validation}


def _plot(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for pid, c in curves.items():
        ax.plot(c["curve"]["nsigma"], c["curve"]["fpr"], "o-", ms=3, label=f"{pid} FPR")
        ax.plot(c["curve"]["nsigma"], c["curve"]["fpr_floor"], ":", lw=1, label=f"{pid} resolution 1/n_pos")
    ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("peak significance threshold [sigma]")
    ax.set_ylabel("false-positive rate (negative image)")
    ax.legend(fontsize=7)
    ax.set_title("Stage 3c: false-positive curve")
    fig.tight_layout()
    fig.savefig(out / "fpr_curve.png", dpi=110)
    plt.close(fig)
