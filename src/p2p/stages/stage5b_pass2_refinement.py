"""Stage 5b - Deterministic Pass-2 refinement.

Purpose : physics-calibrated morphology for the galaxy track.
Method  : the empirical PSF from Stage 5a is fed back into the SAME statmorph
          measurement (same image, same segmentation, same gain) and the galaxy
          track is re-measured exactly once:
            psf_fit    (default) the Sersic model is convolved with the ePSF, so
                       n and r_eff are PSF-corrected; non-parametric indices are
                       re-measured on the same image;
            deconvolve Richardson-Lucy deconvolution (10 fixed iterations)
                       inside the dilated footprint, then statmorph.
          Split tags and cluster IDs are hashed before and after: Pass 2 is not
          allowed to change them.
Output  : features_pass2.csv (psf_pass = 2, galaxy rows only - load_objects()
          lets these overwrite the Pass-1 values), morph_delta_pass.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_objects, save_json
from ..morphology import FEATURES, run_parallel
from ..utils import galaxy_track

FROZEN = ["object_id", "split_tag", "cluster_semantic", "cluster_id_gmm", "is_transitional", "anomaly_flag"]


def _hash(df):
    return int(pd.util.hash_pandas_object(df[FROZEN].astype(str), index=False).sum())


def run(ctx: Ctx):
    out = ctx.stage_dir("5b")
    mode = ctx.cfg["morphology"]["pass2_mode"]
    df = load_objects(ctx, ["sources", "pass1", "clusters", "segregation"])
    before = _hash(df)
    gal = df[galaxy_track(df) & (df.snr >= ctx.cfg["morphology"]["min_snr"]).to_numpy()]
    psf = {p: np.load(ctx.require("5a", f"epsf_{p}.npy")) for p in gal.pointing.unique()}
    ctx.log.info(f"[Stage 5b] Pass-2 ({mode}) on {len(gal)} galaxies with {ctx.n_jobs} worker(s) ...")
    res = run_parallel(ctx, ctx.require("1d", "cutouts.npy"), ctx.require("1c", "segstamps.npy"),
                       gal.cutout_index.tolist(), gal.exptime.tolist(), [psf[p] for p in gal.pointing], mode=mode)
    p2 = pd.DataFrame(res)[FEATURES]
    p2.insert(0, "object_id", gal.object_id.to_numpy())
    p2["psf_pass"] = 2
    p2["pass2_mode"] = mode
    p2.to_csv(out / "features_pass2.csv", index=False)
    assert _hash(df) == before, "Pass 2 changed a frozen column (split / cluster)!"

    p1 = gal.set_index("object_id")[FEATURES]
    d = (p2.set_index("object_id")[FEATURES] - p1).add_suffix("_delta")
    d.reset_index().to_csv(out / "morph_delta_pass.csv", index=False)
    med = d[[c for c in d.columns if c.split("_delta")[0] in ("gini", "m20", "concentration", "asymmetry",
                                                               "sersic_n", "sersic_rhalf", "r_half")]].median()
    save_json({"mode": mode, "galaxies": len(gal), "median_delta": med.round(4).to_dict(),
               "frozen_columns_unchanged": True}, out / "summary.json")
    ctx.log.info("[Stage 5b] median Pass-2 minus Pass-1:\n" + med.round(4).to_string())
    return {"galaxies": len(gal)}
