"""Stage 7c - Generalisation disentanglement.

Purpose : report true degradation, not a raw score drop.
Method  : true degradation = Taxonomy Ceiling - Cross-Catalogue Macro-F1.
          The in-domain test score is recomputed on the same coarse taxonomy so
          all three numbers share one scale. The external sample is also split
          into SNR tertiles and reported per split (train, dev, test, buffer; external once unsealed), and the
          covariate shift (SNR, size, morphology) between the GZH training set and the CANDELS-only galaxies is
          listed.
Output  : generalisation_report.json, generalisation_gap.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..context import Ctx, StageSkipped, load_json, save_json
from ..features import galaxy_table, supervised_splits
from ..nets import MORPH_CLASSES
from .stage7a_taxonomy import GZH_TO_COMMON


def run(ctx: Ctx):
    out = ctx.stage_dir("7c")
    if not ctx.path("7b", "cross_catalogue_metrics.json").exists():
        raise StageSkipped("Stage 7b has no result.")
    ceil = load_json(ctx.require("7a", "taxonomy_ceiling.json"))["ceiling_macro_f1"]
    xcat = load_json(ctx.require("7b", "cross_catalogue_metrics.json"))["xcat_macro_f1"]
    names = np.array(MORPH_CLASSES)
    f1 = lambda a, b: float(f1_score(a, b, labels=sorted(set(a) | set(b)), average="macro", zero_division=0))
    if ctx.path("6a", "test_predictions.csv").exists():
        tp = pd.read_csv(ctx.path("6a", "test_predictions.csv"))
        indomain = round(f1(pd.Series(names[tp.y_true]).map(GZH_TO_COMMON), pd.Series(names[tp.y_pred]).map(GZH_TO_COMMON)), 4)
    else:
        indomain = None                     # no labelled in-field test galaxies (apply mode without GZH)
    xp = pd.read_csv(ctx.require("7b", "crosscat_predictions.csv"))
    tert = pd.qcut(xp.snr, 3, labels=["low SNR", "mid SNR", "high SNR"]) if len(xp) >= 9 else None
    by_snr = {str(k): round(f1(g.k15_common, g.pred_common), 4) for k, g in xp.groupby(tert, observed=True)} if tert is not None else {}
    ext = galaxy_table(ctx).set_index("object_id").loc[xp.object_id]
    SHIFT_COLS = ("snr", "mag", "r_half", "gini", "asymmetry", "concentration")
    num_median = lambda s: float(pd.to_numeric(s, errors="coerce").median())   # robust to text-typed columns
    from ..modes import card, is_apply
    if is_apply(ctx):           # the training set is the bundle's (another field): use its stored medians
        tmed = card(ctx).get("train_feature_medians", {})
    else:
        train = supervised_splits(ctx)["train"]
        tmed = {c: num_median(train[c]) for c in SHIFT_COLS if c in train.columns}
    shift = {c: {"gzh_train_median": round(float(tmed[c]), 3), "external_median": round(num_median(ext[c]), 3)}
             for c in SHIFT_COLS if c in tmed and c in ext}
    rep = {"indomain_macro_f1_coarse": indomain, "taxonomy_ceiling_macro_f1": ceil, "cross_catalogue_macro_f1": xcat,
           "true_degradation": round(ceil - xcat, 4),
           "naive_drop": round(indomain - xcat, 4) if indomain is not None else None,
           "label_scheme_component": round(indomain - ceil, 4) if indomain is not None else None,
           "xcat_macro_f1_by_snr_tertile": by_snr,
           "covariate_shift": shift,
           "xcat_macro_f1_by_split": load_json(ctx.require("7b", "cross_catalogue_metrics.json")).get("by_split", {}),
           "ceiling_by_split": load_json(ctx.require("7a", "taxonomy_ceiling.json")).get("by_split", {}),
           "phase7_splits": load_json(ctx.require("7a", "taxonomy_ceiling.json")).get("phase7_splits", [])}
    if rep["phase7_splits"] == ["all"]:
        rep["note"] = ("apply mode: the frozen model was trained on another field, so every galaxy here is "
                       "out-of-field; the gap mixes the field change (depth, PSF, bands) and the label-scheme change")
    elif "external" not in rep["phase7_splits"]:
        rep["note"] = ("external galaxies sealed: Phase 7 used the train, dev, test and buffer galaxies (CANDELS-only "
                       "galaxies never had a training label); depth and PSF match training, so the gap measures the "
                       "label-scheme change and the fainter CANDELS-only galaxies, not a new field")
    save_json(rep, out / "generalisation_report.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    vals = [indomain if indomain is not None else 0.0, ceil, xcat]
    ax.bar(["In-domain\n(coarse)", "Taxonomy\nceiling", "Cross-\ncatalogue"], vals, color=["#9AA3B5", "#D3D9E5", "#2E5496"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, "n/a" if (i == 0 and indomain is None) else f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro-F1 (shared taxonomy)")
    ax.set_title(f"Stage 7c: true degradation = {ceil - xcat:+.3f}")
    fig.tight_layout()
    fig.savefig(out / "generalisation_gap.png", dpi=110)
    plt.close(fig)
    ctx.log.info(f"[Stage 7c] {rep}")
    return {"true_degradation": rep["true_degradation"]}
