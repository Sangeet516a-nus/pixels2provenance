"""Stage 7a - Taxonomy alignment and the human-agreement ceiling.

Purpose : separate label-scheme differences from model failure.
Method  : galaxies in the EXTERNAL (CANDELS) pointings - or, with a single field,
          in the held-out TEST split (partition.phase7_region: auto) - are matched to the
          Kartaltepe et al. (2015) expert catalogue. Both schemes are mapped to
          a shared coarse taxonomy (Spheroid, Disk, Irregular, Unclassifiable;
          face-on and edge-on discs both become Disk; expert point sources are
          dropped). On galaxies labelled by BOTH GZH and the experts, the
          alignment matrix and the Macro-F1 of GZH vs expert labels give the
          Taxonomy Ceiling - the best any GZH-trained model could score.
          Leakage guards: no evaluated object may lie within 1" of an object
          on the training side (train/dev, plus test when external fields exist).
Output  : candels_objects.csv, taxonomy_alignment_matrix.csv, taxonomy_ceiling.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from ..catalogs import load_candels
from ..context import Ctx, StageSkipped, load_json, save_json
from ..features import galaxy_table
from ..utils import match_with_offset, sky_crossmatch

K15 = ("spheroid", "disk", "irregular", "point_source", "unclassifiable")
COMMON = ("Spheroid", "Disk", "Irregular", "Unclassifiable")
GZH_TO_COMMON = {"Smooth": "Spheroid", "FaceOnSpiral": "Disk", "EdgeOn": "Disk", "Irregular": "Irregular",
                 "Ambiguous": "Unclassifiable"}
K15_TO_COMMON = {"spheroid": "Spheroid", "disk": "Disk", "irregular": "Irregular", "unclassifiable": "Unclassifiable",
                 "point_source": None}


def phase7_region(ctx: Ctx, df) -> str:
    """Which galaxies Phase 7 evaluates on. 'external' = separate CANDELS pointings (proposal design).
    With ONE field there are none, so 'auto' falls back to the TEST split: those galaxies were never
    used for training, tuning or calibration (sky-region split), so the evaluation is still leakage-free;
    it measures transfer to the expert label scheme on held-out sky, with the same depth and PSF."""
    want = str(ctx.cfg["partition"].get("phase7_region", "auto")).lower()
    if want in ("external", "test"):
        return want
    return "external" if (df.split_tag == "external").any() else "test"


def run(ctx: Ctx):
    out = ctx.stage_dir("7a")
    df = galaxy_table(ctx)
    region = phase7_region(ctx, df)
    ext = df[df.split_tag == region].copy()
    if ext.empty:
        raise StageSkipped(f"No galaxies in the '{region}' region for Phase 7. Add a CANDELS field to data/raw and "
                           "list a filename/TARGNAME substring in partition.external_patterns.")
    if region == "test":
        ctx.log.info("[Stage 7a] no external pointing: Phase 7 uses the held-out TEST split "
                     "(partition.phase7_region: auto). Galaxies there were never trained on.")
    metas = list(load_json(ctx.require("1a", "manifest.json")).values())
    k15 = load_candels(ctx, metas)
    if k15 is None or k15.empty:
        raise StageSkipped("No CANDELS (Kartaltepe et al. 2015) table in data/catalogs covering the field.")
    inner = df[df.split_tag.isin([s for s in ("train", "dev", "test") if s != region])]
    _, _, leak = sky_crossmatch(ext.ra, ext.dec, inner.ra, inner.dec, 1.0)
    assert not leak.any(), f"{region} object within 1 arcsec of a training-side object!"

    idx, sep, hit, off = match_with_offset(ext.ra, ext.dec, k15.ra, k15.dec, 0.5)
    F = k15[list(K15)].to_numpy(float)
    main = np.array(K15)[np.nanargmax(np.nan_to_num(F, nan=-1), axis=1)]
    ext["k15_class"] = np.where(hit, main[np.clip(idx, 0, None)], None)
    ext["k15_common"] = ext.k15_class.map(K15_TO_COMMON)
    ext["k15_merger_frac"] = np.where(hit, k15.merger.to_numpy(float)[np.clip(idx, 0, None)], np.nan)
    ext["gzh_common"] = ext.gz_label.map(GZH_TO_COMMON)
    ext["in_gzh_overlap"] = ~ext.is_pseudo_label & ext.k15_common.notna()
    keep = ["object_id", "ra", "dec", "snr", "mag", "k15_class", "k15_common", "k15_merger_frac", "gz_label",
            "gzh_common", "in_gzh_overlap"]
    ext[keep].to_csv(out / "candels_objects.csv", index=False)

    ov = ext[ext.in_gzh_overlap & ext.gzh_common.notna()]
    if len(ov) < 5:
        raise StageSkipped(f"Only {len(ov)} galaxies are labelled by both GZH and CANDELS - no ceiling possible.")
    mat = pd.crosstab(pd.Series(ov.gzh_common.to_numpy(), name="GZH (mapped)"),
                      pd.Series(ov.k15_common.to_numpy(), name="CANDELS expert"))
    mat.to_csv(out / "taxonomy_alignment_matrix.csv")
    present = sorted(set(ov.k15_common) | set(ov.gzh_common))        # only classes that occur are scored
    ceil = {"ceiling_macro_f1": round(float(f1_score(ov.k15_common, ov.gzh_common, labels=present,
                                                     average="macro", zero_division=0)), 4),
            "classes_scored": present,
            "cohen_kappa": round(float(cohen_kappa_score(ov.k15_common, ov.gzh_common)), 4),
            "n_overlap": int(len(ov)), "n_external_galaxies": int(len(ext)), "phase7_region": region,
            "n_k15_matched": int(hit.sum()), "k15_offset_arcsec": off}
    save_json(ceil, out / "taxonomy_ceiling.json")
    ctx.log.info("[Stage 7a] alignment matrix (overlap set):\n" + mat.to_string())
    ctx.log.info(f"[Stage 7a] {ceil}")
    return ceil
