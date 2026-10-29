"""Stage 4d - Controlled ablation: what does each part of the fusion add? (Objective 3)

Rows    : every row is XGBoost with the Stage 4b settings, cross-validated on the SAME 5 folds (number of trees
          chosen from its own out-of-fold curve), on a subset of the 1,806 features:
            ResNet-18 only (512) | EfficientNet-B0 only (1,280) | physical only (14) | CNN features only (1,792)
            | full fusion (1,806, Stage 4b) | full fusion + calibration (the final model, Stage 4c)
            | full fusion with 128 px input (no resize; supervised.ablation_input_size)
            | the previous three-model ensemble (supervised.compare_previous_ensemble, same galaxies).
          Star segregation vs Gaia: the Stage 2c baseline against the learned clustering (Stage 2d).
Tier    : full fusion minus the better single source (CNN features only vs physical only):
          >= 0 Minimum, >= 0.03 Target, >= 0.05 Stretch.
Output  : ablation_table.csv, ablation.png, summary.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_json, save_json
from ..utils import expected_calibration_error


def run(ctx: Ctx):
    from ..context import StageSkipped
    from ..modes import is_apply
    if is_apply(ctx):
        raise StageSkipped("apply mode: the models come from a bundle - the ablation belongs to the training run")
    from ..features import supervised_splits
    from ..inference import block_feature_weights, block_slices, extract_cnn_features, fusion_matrix, xgb_cv
    from ..nets import MORPH_CLASSES, softmax_np
    from ..supervised_cv import choose_epoch, curve_metrics, pooled, summarise
    from .stage4b_xgboost_fusion import tree_grid
    cfg = ctx.cfg["supervised"]
    out = ctx.stage_dir("4d")
    meta = load_json(ctx.require("4b", "fusion_meta.json"))
    physical = meta["physical_columns"]
    w = np.array([meta["ensemble_weights"][b] for b in ("resnet18", "efficientnet_b0", "physical")])
    params, grid = meta.get("xgboost") or {}, tree_grid(cfg)

    # the same galaxies and folds as Stage 4b
    pool = pooled(supervised_splits(ctx))
    fdf = pd.read_csv(ctx.require("4b", "cv_folds.csv")).set_index("object_id")
    fold_id = fdf.fold.reindex(pool.object_id).to_numpy()
    if np.isnan(fold_id.astype(float)).any():
        raise RuntimeError("Stage 4b cv_folds.csv does not match the current labelled galaxies - rerun Stage 4b.")
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in sorted(set(fold_id))]
    y = pool.gz_label_id.astype(int).to_numpy()
    ym = pool.gz_merger_flag.fillna(0).astype(int).to_numpy()
    X = fusion_matrix(ctx, pool, physical)
    sl = block_slices(len(physical))
    fw_all = block_feature_weights(w, len(physical))
    rows = []

    def add(name, rep, P, pm=None, n_trees=None, n_features=None):
        s = summarise(y, P, folds, ym, pm)
        rows.append({"task": "morphology (out-of-fold, train + dev)", "configuration": name, "representation": rep,
                     "n_features": n_features, "trees": n_trees,
                     "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "macro_f1_ci_low": s["macro_f1_95ci"][0],
                     "macro_f1_ci_high": s["macro_f1_95ci"][1], "fold_sd": s["fold_sd"],
                     "balanced_acc": s["balanced_acc"], "log_loss": s["nll"],
                     "merger_pr_auc": s.get("merger_pr_auc", np.nan), "ece": round(expected_calibration_error(P, y), 4)})
        ctx.log.info(f"  [ablation] {name:52s} accuracy {s['accuracy']:.3f}  Macro-F1 {s['macro_f1']:.3f}  "
                     f"log-loss {s['nll']:.3f}")

    def cv_subset(cols, XX=None):
        XX = X if XX is None else XX
        fw = fw_all[cols]
        oof, _ = xgb_cv(XX[:, cols], y, folds, grid, len(MORPH_CLASSES), ctx.seed, params, fw / fw.sum())
        k = choose_epoch(curve_metrics(y, oof, folds), 1, str(cfg.get("selection_metric", "log_loss")))[0] - 1
        return oof[k], grid[k]

    r = lambda b: np.arange(sl[b].start, sl[b].stop)
    for name, rep, cols in (("ResNet-18 features only", "resnet18", r("resnet18")),
                            ("EfficientNet-B0 features only", "efficientnet_b0", r("efficientnet_b0")),
                            ("Physical features only (classical)", "physical", r("physical")),
                            ("CNN features only (ResNet-18 + EfficientNet-B0)", "resnet18+efficientnet_b0",
                             np.r_[r("resnet18"), r("efficientnet_b0")])):
        P, n = cv_subset(cols)
        add(name, rep, P, n_trees=n, n_features=int(len(cols)))
    d = np.load(ctx.require("4b", "oof_fusion.npz"))
    add("Full fusion (Stage 4b)", "resnet18+efficientnet_b0+physical", d["morph"], d["merger"], meta["trees"],
        int(X.shape[1]))
    T = load_json(ctx.require("4c", "temperature.json"))
    add("Full fusion + calibration (final model)", "resnet18+efficientnet_b0+physical",
        softmax_np(np.log(np.clip(d["morph"], 1e-12, 1)) / T["T_morph"]), d["merger"], meta["trees"], int(X.shape[1]))

    small = int(cfg.get("ablation_input_size") or 0)
    if small and small != int(meta.get("input_size", 224)):
        ctx.log.info(f"[Stage 4d] CNN features at {small} px for the {len(pool)} labelled galaxies ...")
        X2 = X.copy()
        X2[:, :1792] = _features_at(ctx, pool.cutout_index.to_numpy(), small, extract_cnn_features)
        P, n = cv_subset(np.arange(X.shape[1]), X2)
        add(f"Full fusion, {small} px input (no resize)", "resnet18+efficientnet_b0+physical", P, n_trees=n,
            n_features=int(X.shape[1]))

    prev = cfg.get("compare_previous_ensemble")
    pf = ctx.root / str(prev) / "stage4b_ensemble" / "oof_members.npz" if prev else None
    if pf is not None and pf.exists():
        o = np.load(pf)
        pos = {int(v): i for i, v in enumerate(o["object_id"])}
        rows_prev = np.array([pos.get(int(v), -1) for v in pool.object_id])
        if (rows_prev >= 0).all():
            add("Previous three-model ensemble (dual-branch + EfficientNet + XGBoost)", "ensemble",
                o["ensemble"][rows_prev], o["merger"][rows_prev])
            rows[-1]["note"] = f"from {prev}: weighted vote of three classifiers, out-of-fold"
        else:
            ctx.log.info(f"[Stage 4d] {pf} covers {int((rows_prev >= 0).sum())} of {len(pool)} galaxies - "
                         "different labelled set, row skipped")
    elif prev:
        ctx.log.info(f"[Stage 4d] supervised.compare_previous_ensemble: {pf} not found - row skipped")

    audit = load_json(ctx.require("2d", "semantic_audit.json"))
    for name, key in (("Baseline segregator (Stage 2c, no Gaia leg)", "baseline_star_vs_gaia"),
                      ("CAE + GMM/HDBSCAN (Stage 2d)", "learned_star_vs_gaia")):
        m = audit[key]
        rows.append({"task": "star segregation vs Gaia", "configuration": name, "star_precision": m["precision"],
                     "star_recall": m["recall"], "star_f1": m["f1"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ablation_table.csv", index=False)
    f = tab.set_index("configuration").macro_f1
    delta = float(f["Full fusion (Stage 4b)"] - max(f["CNN features only (ResNet-18 + EfficientNet-B0)"],
                                                     f["Physical features only (classical)"]))
    tier = "Stretch" if delta >= 0.05 else "Target" if delta >= 0.03 else "Minimum" if delta >= 0 else "Below minimum"
    save_json({"fusion_delta_macro_f1": round(delta, 4), "tier": tier,
               "scored_on": "out-of-fold predictions, pooled train + dev galaxies, Stage 4b folds"}, out / "summary.json")
    _plot(tab[tab.task.str.startswith("morphology")], out)
    ctx.log.info("[Stage 4d]\n" + tab.drop(columns=["task"]).to_string(index=False))
    ctx.log.info(f"[Stage 4d] full fusion minus the better single source = {delta:+.4f} -> {tier}")
    return {"fusion_delta": round(delta, 4), "tier": tier}


def _features_at(ctx, cutout_index, size, extract):
    """CNN features with another input size (the config is changed for this call only)."""
    old = ctx.cfg["supervised"].get("input_size", 224)
    ctx.cfg["supervised"]["input_size"] = size
    try:
        return extract(ctx, cutout_index)
    finally:
        ctx.cfg["supervised"]["input_size"] = old


def _plot(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = t.sort_values("macro_f1")
    col = ["#eb6834" if "final model" in c else "#2E5496" if c.startswith("Full fusion") else
           "#9AA3B5" if "only" in c else "#D3D9E5" for c in t.configuration]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.2))
    ax.barh(t.configuration, t.macro_f1, color=col)
    for i, (v, a) in enumerate(zip(t.macro_f1, t.accuracy)):
        ax.text(v + 0.01, i, f"Macro-F1 {v:.3f} | acc {a:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("out-of-fold Macro-F1 (pooled train + dev, same 5 folds)")
    ax.set_title("Stage 4d ablation - orange: final model; blue: fusion; grey: single feature sources", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=110)
    plt.close(fig)
