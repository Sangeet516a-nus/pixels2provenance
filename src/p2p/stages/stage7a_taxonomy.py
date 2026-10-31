"""Stage 7a - Taxonomy alignment and the human-agreement ceiling.

Purpose : separate label-scheme differences from model failure.
Method  : galaxies of EVERY scored split (train, dev, test, buffer; external too once evaluation.seal_external is
          false) are matched to the Kartaltepe et al. (2015) expert catalogue (per-pointing offset, 0.5"). Both
          schemes are mapped to a shared coarse taxonomy (Spheroid, Disk, Irregular, Unclassifiable; face-on and
          edge-on discs both become Disk; expert point sources are dropped). On galaxies labelled by BOTH GZH and the
          experts, the alignment matrix and the Macro-F1 of GZH vs expert labels give the Taxonomy Ceiling - the best
          any GZH-trained model could score. The ceiling compares two HUMAN catalogues (no model is involved), so it
          may use galaxies of any split; it is reported overall and per split. Pointings without CANDELS coverage
          (GOODS-North) simply have no matches.
Output  : candels_objects.csv (with split_tag), taxonomy_alignment_matrix.csv, taxonomy_ceiling.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from ..catalogs import load_candels
from ..context import Ctx, StageSkipped, load_json, save_json
from ..features import galaxy_table
from ..utils import match_with_offset

K15 = ("spheroid", "disk", "irregular", "point_source", "unclassifiable")
COMMON = ("Spheroid", "Disk", "Irregular", "Unclassifiable")
GZH_TO_COMMON = {"Smooth": "Spheroid", "FaceOnSpiral": "Disk", "EdgeOn": "Disk", "Irregular": "Irregular",
                 "Ambiguous": "Unclassifiable"}
K15_TO_COMMON = {"spheroid": "Spheroid", "disk": "Disk", "irregular": "Irregular", "unclassifiable": "Unclassifiable",
                 "point_source": None}


def phase7_galaxies(ctx: Ctx, df):
    """The galaxies Phase 7 uses: every scored split (external only when unsealed); apply mode: all."""
    from ..modes import is_apply
    from ..sealing import scored_splits
    if is_apply(ctx):
        return df.copy(), ["all"]
    splits = scored_splits(ctx)
    return df[df.split_tag.fillna("buffer").isin(splits)].copy(), splits


def run(ctx: Ctx):
    out = ctx.stage_dir("7a")
    df = galaxy_table(ctx)
    ext, splits = phase7_galaxies(ctx, df)
    ext["split_tag"] = ext.split_tag.fillna("buffer")
    from ..sealing import external_sealed
    ctx.log.info(f"[Stage 7a] Phase 7 uses the splits {splits} ({len(ext):,} galaxies)"
                 + ("; external galaxies excluded while evaluation.seal_external is true" if external_sealed(ctx) else ""))
    if ext.empty:
        raise StageSkipped("No galaxies for Phase 7.")
    metas = list(load_json(ctx.require("1a", "manifest.json")).values())
    k15 = load_candels(ctx, metas)
    if k15 is None or k15.empty:
        raise StageSkipped("No CANDELS (Kartaltepe et al. 2015) table in data/catalogs covering the field.")

    idx, sep, hit, off = match_with_offset(ext.ra, ext.dec, k15.ra, k15.dec, 0.5, groups=ext.pointing.to_numpy())
    F = k15[list(K15)].to_numpy(float)
    main = np.array(K15)[np.nanargmax(np.nan_to_num(F, nan=-1), axis=1)]
    ext["k15_class"] = np.where(hit, main[np.clip(idx, 0, None)], None)
    ext["k15_common"] = ext.k15_class.map(K15_TO_COMMON)
    ext["k15_merger_frac"] = np.where(hit, k15.merger.to_numpy(float)[np.clip(idx, 0, None)], np.nan)
    ext["gzh_common"] = ext.gz_label.map(GZH_TO_COMMON)
    ext["in_gzh_overlap"] = ~ext.is_pseudo_label & ext.k15_common.notna()
    keep = ["object_id", "split_tag", "pointing", "ra", "dec", "snr", "mag", "k15_class", "k15_common", "k15_merger_frac", "gz_label",
            "gzh_common", "in_gzh_overlap"]
    ext[keep].to_csv(out / "candels_objects.csv", index=False)

    ov = ext[ext.in_gzh_overlap & ext.gzh_common.notna()]
    if len(ov) < 5:
        raise StageSkipped(f"Only {len(ov)} galaxies are labelled by both GZH and CANDELS - no ceiling possible.")
    mat = pd.crosstab(pd.Series(ov.gzh_common.to_numpy(), name="GZH (mapped)"),
                      pd.Series(ov.k15_common.to_numpy(), name="CANDELS expert"))
    mat.to_csv(out / "taxonomy_alignment_matrix.csv")
    present = sorted(set(ov.k15_common) | set(ov.gzh_common))        # only classes that occur are scored

    def _ceil(o):
        pr = sorted(set(o.k15_common) | set(o.gzh_common))
        return {"n_overlap": int(len(o)), "ceiling_macro_f1": round(float(f1_score(o.k15_common, o.gzh_common,
                labels=pr, average="macro", zero_division=0)), 4)} if len(o) >= 5 else {"n_overlap": int(len(o))}
    ceil = {"ceiling_macro_f1": round(float(f1_score(ov.k15_common, ov.gzh_common, labels=present,
                                                     average="macro", zero_division=0)), 4),
            "classes_scored": present,
            "cohen_kappa": round(float(cohen_kappa_score(ov.k15_common, ov.gzh_common)), 4),
            "n_overlap": int(len(ov)), "n_phase7_galaxies": int(len(ext)), "phase7_splits": splits,
            "by_split": {k: _ceil(g) for k, g in ov.groupby("split_tag")},
            "k15_matched_by_split": {k: int(v) for k, v in pd.Series(hit).groupby(ext.split_tag.to_numpy()).sum().items()},
            "n_k15_matched": int(hit.sum()), "k15_offset_arcsec": off}
    save_json(ceil, out / "taxonomy_ceiling.json")
    ctx.log.info("[Stage 7a] alignment matrix (overlap set):\n" + mat.to_string())
    ctx.log.info(f"[Stage 7a] {ceil}")
    return ceil
