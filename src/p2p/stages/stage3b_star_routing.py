"""Stage 3b - Stars: calibration routing.

Purpose : point sources skip morphology classification and become PSF tracers.
Method  : candidates are the star track PLUS every Gaia-matched object (a Gaia
          star mis-assigned to another track is still a star). A candidate is
          routed to the empirical-PSF builder if it is a Gaia star or on the
          bright stellar locus of Stage 2d (point-like, SNR >=
          baseline.star_locus_min_snr, confirmed in a second band), and it has SNR >= 20, is not saturated (no flat-topped
          core in the detection band), has no other segment within
          psf_match.star_isolation_px (5 px) of its centre (Stage 5a leaves
          neighbour and cosmic-ray pixels out of the stack) and is not at an edge. Only Gaia stars are used
          whenever at least 5 of them in its pointing pass those cuts (in deep
          fields most Gaia stars saturate).
Output  : star_routing.csv (one row per star with each test), routing_summary.json
"""
from __future__ import annotations

import numpy as np

from ..context import Ctx, load_json, load_objects, save_json
from ..utils import as_bool
from .stage1d_psf_match import flat_top


def run(ctx: Ctx, min_snr=20.0, min_epsf_stars=5):
    iso_radius = ctx.cfg["psf_match"].get("star_isolation_px", 5)   # Stage 5a leaves neighbour pixels out
    out = ctx.stage_dir("3b")
    df = load_objects(ctx, ["sources", "pass1", "gaia", "evidence", "segregation"])
    tol = ctx.cfg["baseline"]["point_source_tolerance"]
    bands = load_json(ctx.require("1a", "pointings.json"))["bands"]
    raw = np.load(ctx.require("1c", "cutouts_raw.npy"), mmap_mode="r")
    segs = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    stars = df[((df.cluster_semantic == "Star") | as_bool(df.gaia_hit)) & (df.anomaly_flag == 0)].copy()
    S = segs.shape[1]
    yy, xx = np.mgrid[:S, :S]
    near = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2) <= iso_radius
    det_i = stars.detection_band.map(bands.index).to_numpy() if len(stars) else []
    stars["epsf_snr_ok"] = stars.snr >= min_snr
    stars["epsf_unsaturated"] = [not flat_top(raw[i, d]) for i, d in zip(stars.cutout_index, det_i)]
    stars["epsf_isolated"] = [not np.any(segs[i][near] == 2) for i in stars.cutout_index]
    stars["epsf_not_edge"] = ~as_bool(stars.edge_flag)
    stars["epsf_unsaturated"] &= ~as_bool(stars.saturated) if "saturated" in stars else True
    stars["epsf_point_like"] = (stars.fwhm_over_psf.astype(float) - 1.0).abs() <= tol
    stars["epsf_two_band"] = stars.cr_candidate.map(lambda v: str(v).strip().lower() in ("false", "0", "0.0")) \
        if "cr_candidate" in stars else True
    gaia = as_bool(stars.gaia_hit)
    locus = as_bool(stars.stellar_locus) if "stellar_locus" in stars else (stars.epsf_point_like & stars.epsf_two_band)
    stars["epsf_star_evidence"] = gaia | (locus & stars.epsf_point_like)
    ok = (stars.epsf_star_evidence & stars.epsf_snr_ok & stars.epsf_unsaturated & stars.epsf_isolated
          & stars.epsf_not_edge)
    # Gaia confirmation is required only when enough confirmed stars survive the cuts: in a deep field
    # most Gaia stars (G < 21) saturate a long ACS exposure, and Stage 5a needs >= 5 stars per pointing
    gaia_required = {}
    for pid, idx in stars.groupby("pointing").groups.items():
        with_gaia = ok.loc[idx] & gaia.loc[idx]
        gaia_required[pid] = bool(with_gaia.sum() >= min_epsf_stars)
        if gaia_required[pid]:
            ok.loc[idx] = with_gaia
        elif gaia.loc[idx].any():
            ctx.log.info(f"[Stage 3b] {pid}: only {int(with_gaia.sum())} Gaia-confirmed clean stars - using all "
                         f"{int(ok.loc[idx].sum())} clean stars of the star track for the ePSF")
    stars["used_in_epsf"] = ok
    cols = ["object_id", "pointing", "snr", "gaia_hit", "cluster_semantic", "epsf_point_like", "epsf_two_band",
            "epsf_star_evidence", "epsf_snr_ok", "epsf_unsaturated", "epsf_isolated", "epsf_not_edge", "used_in_epsf"]
    stars[cols].to_csv(out / "star_routing.csv", index=False)
    per = stars.groupby("pointing").used_in_epsf.sum().astype(int).to_dict()
    save_json({"stars": len(stars), "routed": int(ok.sum()), "per_pointing": per,
               "gaia_required_per_pointing": gaia_required}, out / "routing_summary.json")
    ctx.log.info(f"[Stage 3b] candidates {len(stars)} (star track + Gaia) | Gaia {int(gaia.sum())} | point-like "
                 f"{int(stars.epsf_point_like.sum())} | confirmed in 2 bands {int(stars.epsf_two_band.sum())} | "
                 f"SNR ok {int(stars.epsf_snr_ok.sum())} | unsaturated "
                 f"{int(stars.epsf_unsaturated.sum())} | isolated {int(stars.epsf_isolated.sum())} | routed to ePSF "
                 f"{int(ok.sum())} {per}")
    return {"routed": int(ok.sum())}
