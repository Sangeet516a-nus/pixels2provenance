"""Stage 3a - Galaxies: Galaxy Zoo: Hubble label anchoring.

Purpose : attach crowd-sourced morphology labels to the galaxy track.
Method  : galaxy positions are matched one-to-one to GZH within 0.5" after the
          median catalogue-to-image offset is removed. GZH vote fractions are
          conditional on the earlier answers of the decision tree, so:
          (rule of 31 Oct; GZH itself holds vote fractions only, the classes are ours)
            Smooth        f_smooth >= 0.6, or f_smooth >= 0.4 and f_smooth >= f_features
            featured      f_features >= 0.4 (and not Smooth), then, in this order:
              EdgeOn        f_edgeon >= 0.4
              FaceOnSpiral  f_spiral >= 0.4
              Irregular     f_clumpy >= 0.5, or f_odd >= 0.4 and max(irregular, disturbed) >= 0.4
            Ambiguous     everything else
          merger flag: f_odd >= 0.4 and f_merger >= 0.5.
          Human agreement = the WEAKEST vote fraction along the path to the
          assigned class; matched galaxies with agreement < 0.5 form the
          Ambiguous Boundary set used by Stage 6b. Unmatched galaxies are kept
          for inference only (is_pseudo_label = True).
Output  : labels.csv, label_summary.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..catalogs import load_gzh
from ..context import Ctx, load_json, load_objects, save_json
from ..nets import MORPH_CLASSES
from ..utils import fmt_offsets, galaxy_track, match_with_offset


def map_labels(g: pd.DataFrame, t: dict) -> pd.DataFrame:
    f = lambda c: g[c].fillna(0).to_numpy(float)
    smooth, feat, edge, spiral = f("smooth"), f("features"), f("edgeon"), f("spiral")
    odd, clumpy = f("odd"), f("clumpy")
    irr_odd = np.maximum(f("odd_irregular"), f("odd_disturbed"))
    irregular_votes = np.maximum(clumpy, np.where(odd >= t["odd"], irr_odd, 0.0))
    lab = np.full(len(g), 4)
    is_smooth = (smooth >= t["smooth"]) | ((smooth >= t["smooth_min"]) & (smooth >= feat))
    featured = ~is_smooth & (feat >= t["features"])
    is_edge = featured & (edge >= t["edgeon"])
    is_spiral = featured & ~is_edge & (spiral >= t["spiral"])
    is_irregular = featured & ~is_edge & ~is_spiral & ((clumpy >= t["clumpy"]) | ((odd >= t["odd"]) & (irr_odd >= t["irregular"])))
    lab[is_smooth], lab[is_edge], lab[is_spiral], lab[is_irregular] = 0, 2, 1, 3
    # agreement = weakest link along the decision path
    A = np.column_stack([smooth,
                         np.minimum.reduce([feat, 1 - edge, spiral]),
                         np.minimum(feat, edge),
                         np.minimum.reduce([feat, 1 - edge, 1 - spiral, irregular_votes])])
    A = np.column_stack([A, 1.0 - A.max(1)])
    agree = A[np.arange(len(lab)), lab]
    return pd.DataFrame({"gz_label_id": lab, "gz_label": np.array(MORPH_CLASSES)[lab],
                         "gz_vote_frac": agree.round(4), "human_disagreement": (1 - agree).round(4),
                         "gz_merger_flag": ((odd >= t["odd"]) & (f("odd_merger") >= t["merger"])).astype(int)})


def vote_fractions(g: pd.DataFrame, t: dict) -> np.ndarray:
    """(n, 5) soft targets from the same decision tree as map_labels, in MORPH_CLASSES order: each class gets the
    votes along its path, and whatever no path claims (including star / artifact votes) goes to Ambiguous."""
    f = lambda c: g[c].fillna(0).to_numpy(float)
    smooth, feat, edge, spiral = f("smooth"), f("features"), f("edgeon"), f("spiral")
    irregular = np.maximum(f("clumpy"), np.where(f("odd") >= t["odd"], np.maximum(f("odd_irregular"), f("odd_disturbed")), 0.0))
    v = np.column_stack([smooth, feat * (1 - edge) * spiral, feat * edge, feat * (1 - edge) * (1 - spiral) * irregular])
    v = np.clip(v, 0, None)
    v = v / np.maximum(v.sum(1, keepdims=True), 1.0)            # debiased fractions need not sum to at most 1
    return np.column_stack([v, 1 - v.sum(1)])


def run(ctx: Ctx):
    cfg = ctx.cfg["catalogs"]["gzh"]
    out = ctx.stage_dir("3a")
    df = load_objects(ctx, ["sources", "segregation"])
    metas = list(load_json(ctx.require("1a", "manifest.json")).values())
    from ..modes import is_apply
    gzh = load_gzh(ctx, metas, required=not is_apply(ctx))
    gal = df[galaxy_track(df)].copy()
    if gzh is None or gzh.empty:
        if not is_apply(ctx):
            raise RuntimeError("No Galaxy Zoo: Hubble galaxies cover this field, so the models cannot be trained here. "
                               "Run it in apply mode with a model bundle (supervised.mode: apply, "
                               "supervised.model_bundle: models/<name>; tools/run_field.py does this for you).")
        res = pd.DataFrame({"object_id": gal.object_id.to_numpy(), "gz_label_id": np.nan, "gz_label": None,
                            "gz_vote_frac": np.nan, "human_disagreement": np.nan, "gz_merger_flag": np.nan,
                            "gzh_sep_arcsec": np.nan, "gzh_row": -1, "is_pseudo_label": True, "in_boundary_set": False})
        res.to_csv(out / "labels.csv", index=False)
        summary = {"galaxies": len(gal), "matched": 0, "unmatched": len(gal), "mode": "apply",
                   "note": "no Galaxy Zoo: Hubble coverage - every galaxy is classified by the frozen model bundle"}
        save_json(summary, out / "label_summary.json")
        ctx.log.info(f"[Stage 3a] no GZH galaxies in this field (apply mode): {len(gal)} galaxies, all model-labelled")
        return {"matched": 0}
    # offset measured per pointing (their astrometry differs by up to ~1"; one global offset left GOODS-N-4
    # almost unmatched in the first 8-pointing run)
    idx, sep, hit, off = match_with_offset(gal.ra, gal.dec, gzh.ra, gzh.dec, cfg["match_radius_arcsec"],
                                           estimate=cfg.get("estimate_offset", True), groups=gal.pointing.to_numpy())
    lab = map_labels(gzh, cfg["thresholds"])
    res = pd.DataFrame({"object_id": gal.object_id.to_numpy()})
    for c in lab.columns:
        res[c] = np.where(hit, lab[c].to_numpy()[np.clip(idx, 0, None)], np.nan)
    res["gz_label"] = np.where(hit, lab["gz_label"].to_numpy()[np.clip(idx, 0, None)], None)
    res["gzh_sep_arcsec"] = np.where(hit, sep, np.nan).round(4)
    res["gzh_row"] = np.where(hit, gzh.gzh_row.to_numpy()[np.clip(idx, 0, None)], -1)
    res["is_pseudo_label"] = ~hit
    res["in_boundary_set"] = hit & (res.gz_vote_frac.fillna(1).to_numpy() < cfg["boundary_agreement"])
    res.to_csv(out / "labels.csv", index=False)

    lab_rows = res[~res.is_pseudo_label].merge(gal[["object_id", "split_tag"]], on="object_id")
    table = pd.crosstab(lab_rows.split_tag.fillna("collar"), lab_rows.gz_label)
    from ..sealing import external_sealed
    hide = ["external"] if external_sealed(ctx) else []
    if hide:
        # sealed: the class mix of the sealed galaxies is not shown either - only how many there are
        sealed_rows = [r for r in table.index if r in hide]
        counts = {r: int(table.loc[r].sum()) for r in sealed_rows}
        table = table.drop(index=sealed_rows)
        for r, n in counts.items():
            table.loc[f"{r} (sealed: {n} galaxies)"] = np.nan
    summary = {"galaxies": len(gal), "matched": int(hit.sum()), "unmatched": int((~hit).sum()),
               "offset_arcsec_per_pointing": {k: {"dra": v[0], "ddec": v[1]} for k, v in off.items()},
               "matched_per_pointing": {k: int(v) for k, v in pd.Series(hit).groupby(gal.pointing.to_numpy()).sum().items()},
               "per_split_class": table.to_dict(), "boundary_set": int(res.in_boundary_set.sum()),
               "merger_prevalence_train": float(lab_rows[lab_rows.split_tag == "train"].gz_merger_flag.mean())
               if (lab_rows.split_tag == "train").any() else None}
    save_json(summary, out / "label_summary.json")
    ctx.log.info(f"[Stage 3a] galaxy track {len(gal)} | GZH-matched {int(hit.sum())} | unmatched {int((~hit).sum())} "
                 f"| offset removed per pointing: {fmt_offsets(off)}")
    ctx.log.info("[Stage 3a] labelled galaxies per split / class:\n" + table.to_string())
    return {"matched": int(hit.sum())}
