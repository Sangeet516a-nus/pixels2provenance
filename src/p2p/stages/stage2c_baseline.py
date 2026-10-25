"""Stage 2c - Baseline segregation (+ the direct Gaia DR3 cross-match).

Purpose : a simple, transparent Star / Galaxy / Noise separator - the yardstick
          the learned segregation (Stages 2a-2d) must beat (Objective 2).
Method  : (1) Gaia DR3 match within 0.5" after propagating Gaia positions to the
          HST epoch with proper motions -> Star;
          (2) noise: SNR < 3, a profile SHARPER than the PSF (FWHM < 0.7 x the
              stellar locus - no real source can be sharper; cosmic rays can),
              or a single-band detection (band_consistency < 0.05);
          (3) stellar locus: FWHM within 20% of the stellar locus and a
              CLASS_STAR-like stellarity > 0.5 -> Star;
          (4) everything else -> Galaxy.
          The stellarity is a stand-in for SExtractor's CLASS_STAR built from
          how close the source FWHM is to the PSF FWHM.
Output  : baseline.csv, gaia_match.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..catalogs import load_gaia, propagate_gaia
from ..context import Ctx, load_json, load_objects
from ..utils import noise_evidence, sky_crossmatch


def gaia_crossmatch(ctx: Ctx, df: pd.DataFrame, pts: dict, metas):
    cfg = ctx.cfg["semantic_audit"]
    gaia = load_gaia(ctx, metas)
    out = pd.DataFrame({"object_id": df.object_id, "gaia_hit": False})
    if gaia is None or gaia.empty:
        ctx.log.info("[Stage 2c] no Gaia table (data/catalogs/*gaia*.csv) - Gaia leg disabled")
        return out, False
    ctx.log.info(f"[Stage 2c] Gaia: {len(gaia)} sources in the footprint")
    cols = {k: np.full(len(df), np.nan) for k in ("gaia_sep_arcsec", "parallax", "pmra", "pmdec", "gaia_g", "gaia_teff")}
    sid = np.full(len(df), -1, dtype=np.int64)
    hit_all = np.zeros(len(df), bool)
    for pid, p in pts.items():
        m = (df.pointing == pid).to_numpy()
        if not m.any():
            continue
        g = propagate_gaia(gaia, p["epoch"], ctx.cfg["catalogs"]["gaia"].get("epoch", 2016.0))
        idx, sep, hit = sky_crossmatch(df.ra[m], df.dec[m], g.ra, g.dec, cfg["gaia_radius_arcsec"])
        rows = np.where(m)[0][hit]
        j = idx[hit]
        hit_all[rows] = True
        sid[rows] = pd.to_numeric(g["source_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()[j]
        cols["gaia_sep_arcsec"][rows] = sep[hit]
        for c, gc in (("parallax", "parallax"), ("pmra", "pmra"), ("pmdec", "pmdec"), ("gaia_g", "g_mag"), ("gaia_teff", "teff")):
            cols[c][rows] = g[gc].to_numpy(float)[j]
    out["gaia_hit"] = hit_all
    out["gaia_source_id"] = np.where(hit_all, sid, -1)
    for c, v in cols.items():
        out[c] = v
    return out, True


def run(ctx: Ctx):
    cfg = ctx.cfg["baseline"]
    out = ctx.stage_dir("2c")
    df = load_objects(ctx, ["sources", "pass1"])
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    metas = list(load_json(ctx.require("1a", "manifest.json")).values())
    gm, have_gaia = gaia_crossmatch(ctx, df, pts, metas)
    gm.to_csv(out / "gaia_match.csv", index=False)

    ratio = df.fwhm_over_psf.to_numpy(float)                       # 1 = as compact as a star (Stage 1e)
    stellarity = np.exp(-((ratio - 1.0) ** 2) / (2 * 0.15 ** 2))   # CLASS_STAR-like, 1 = PSF-like
    stellarity = np.where(np.isfinite(stellarity), stellarity, 0.0)
    noise = noise_evidence(df, cfg)
    locus = np.abs(ratio - 1.0) <= cfg["point_source_tolerance"]
    star = locus & (stellarity > cfg["class_star_threshold"]) & ~noise
    label = np.where(noise, "Noise", np.where(star, "Star", "Galaxy")).astype(object)
    label[gm.gaia_hit.to_numpy()] = "Star"
    res = pd.DataFrame({"object_id": df.object_id, "class_star_proxy": stellarity.round(4),
                        "fwhm_over_psf": ratio.round(4), "baseline_semantic": label,
                        "baseline_semantic_nogaia": np.where(noise, "Noise", np.where(star, "Star", "Galaxy"))})
    res.to_csv(out / "baseline.csv", index=False)
    ctx.log.info(f"[Stage 2c] baseline: {pd.Series(label).value_counts().to_dict()} | Gaia matches "
                 f"{int(gm.gaia_hit.sum())} (Gaia leg {'on' if have_gaia else 'off'})")
    return {"baseline_counts": pd.Series(label).value_counts().to_dict(), "gaia_matches": int(gm.gaia_hit.sum())}
