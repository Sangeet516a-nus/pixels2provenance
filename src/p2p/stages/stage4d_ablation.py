"""Stage 4d - Controlled ablation: what does each part of the final model add? (Objective 3)

Rows    : (1) the members of the Stage 4b ensemble, each on its own, from oof_members.npz:
              fine-tuned ResNet-18 | fine-tuned EfficientNet-B0 | the two fine-tuned CNNs averaged
              | full-fusion XGBoost (frozen CNN features + physical) | physical XGBoost member
          (2) the final ensemble (equal-weight average of the members) and the final ensemble + calibration
              (the final model, Stage 4c)
          (3) XGBoost with the Stage 4b fusion settings on a subset of the feature blocks, same 5 folds:
              ResNet-18 features only | EfficientNet-B0 features only | physical only (14) | frozen CNN features only
              | fusion without the CNN PCA | without the label-confidence weights | whole 128 px cutout
              | 128 px input (supervised.ablation_input_size) | previous three-model ensemble (if configured).
          Star segregation vs Gaia: the Stage 2c baseline against the learned clustering (Stage 2d).
Tier    : final ensemble minus the best single source (any images-only row vs physical only):
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
    from ..inference import extract_cnn_features, fusion_matrix, raw_blocks, xgb_cv
    from ..nets import MORPH_CLASSES, softmax_np
    from ..supervised_cv import choose_epoch, curve_metrics, pooled, shape_scores, summarise
    from .stage4b_xgboost_fusion import tree_grid
    cfg = ctx.cfg["supervised"]
    out = ctx.stage_dir("4d")
    meta = load_json(ctx.require("4b", "fusion_meta.json"))
    physical = meta["physical_columns"]
    wd = dict(meta["ensemble_weights"])
    params, grid = meta.get("xgboost") or {}, tree_grid(cfg)
    pca_k = int(meta.get("cnn_pca_components") or 0)

    # the same galaxies and folds as Stage 4b
    pool = pooled(supervised_splits(ctx))
    fdf = pd.read_csv(ctx.require("4b", "cv_folds.csv")).set_index("object_id")
    fold_id = fdf.fold.reindex(pool.object_id).to_numpy()
    if np.isnan(fold_id.astype(float)).any():
        raise RuntimeError("Stage 4b cv_folds.csv does not match the current labelled galaxies - rerun Stage 4b.")
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in sorted(set(fold_id))]
    y = pool.gz_label_id.astype(int).to_numpy()
    ym = pool.gz_merger_flag.fillna(0).astype(int).to_numpy()
    conf = pool.gz_vote_frac.to_numpy(float) if meta.get("label_confidence_weights") else None
    X = fusion_matrix(ctx, pool, physical)
    rb = raw_blocks(len(physical))
    rows = []

    def add(name, rep, P, pm=None, n_trees=None, n_features=None):
        s = summarise(y, P, folds, ym, pm)
        sh = shape_scores(y, P)
        rows.append({"task": "morphology (out-of-fold, train + dev)", "configuration": name, "representation": rep,
                     "n_features": n_features, "trees": n_trees,
                     "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "macro_f1_ci_low": s["macro_f1_95ci"][0],
                     "macro_f1_ci_high": s["macro_f1_95ci"][1], "fold_sd": s["fold_sd"],
                     "balanced_acc": s["balanced_acc"], "log_loss": s["nll"],
                     "merger_pr_auc": s.get("merger_pr_auc", np.nan), "ece": round(expected_calibration_error(P, y), 4),
                     "shape_coverage": sh["coverage"], "shape_accuracy": sh["accuracy"], "shape_macro_f1": sh["macro_f1"]})
        ctx.log.info(f"  [ablation] {name:52s} accuracy {s['accuracy']:.3f}  Macro-F1 {s['macro_f1']:.3f}  "
                     f"log-loss {s['nll']:.3f}")

    def cv_subset(names, XX=None, k=None, cf="same"):
        """XGBoost CV on the blocks `names` of XX (default X), Stage 4b settings unless overridden."""
        XX = X if XX is None else XX
        k = pca_k if k is None else k
        cf = conf if isinstance(cf, str) else cf
        cols, sub, s0 = [], {}, 0
        for b in names:
            a_, z_ = rb[b]
            cols += list(range(a_, z_))
            sub[b] = (s0, s0 + z_ - a_)
            s0 += z_ - a_
        oof, _ = xgb_cv(XX[:, cols], y, folds, grid, len(MORPH_CLASSES), ctx.seed, params,
                        {b: wd[b] for b in names}, sub, k, cf)
        kk = choose_epoch(curve_metrics(y, oof, folds), 1, str(cfg.get("selection_metric", "log_loss")))[0] - 1
        nf = sum(min(k, z_ - a_) if (k and b != "physical") else z_ - a_ for b, (a_, z_) in sub.items())
        return oof[kk], grid[kk], int(nf)

    ALL = ["resnet18", "efficientnet_b0", "physical"]
    for name, rep, names in (("ResNet-18 features only", "resnet18", ["resnet18"]),
                             ("EfficientNet-B0 features only", "efficientnet_b0", ["efficientnet_b0"]),
                             ("Physical features only (classical)", "physical", ["physical"]),
                             ("CNN features only (ResNet-18 + EfficientNet-B0)", "resnet18+efficientnet_b0",
                              ["resnet18", "efficientnet_b0"])):
        P, n, nf = cv_subset(names)
        add(name, rep, P, n_trees=n, n_features=nf)
    d = np.load(ctx.require("4b", "oof_fusion.npz"))
    mo = np.load(ctx.require("4b", "oof_members.npz"))
    members = list(meta.get("ensemble_members") or ["xgb_fusion"])
    labels = {"xgb_fusion": "Full fusion XGBoost (frozen CNN + physical)", "xgb_physical": "Physical XGBoost (member)",
              "resnet18_ft": "Fine-tuned ResNet-18 only", "efficientnet_b0_ft": "Fine-tuned EfficientNet-B0 only"}
    for m in mo.files:
        if m in labels:
            add(labels[m], m, mo[m], n_trees=meta["trees"] if m == "xgb_fusion" else None)
    if "resnet18_ft" in mo.files and "efficientnet_b0_ft" in mo.files:
        add("Fine-tuned CNNs only (average of both)", "resnet18_ft+efficientnet_b0_ft",
            (mo["resnet18_ft"] + mo["efficientnet_b0_ft"]) / 2)
    if meta.get("ensemble_member_weights") and all(m in mo.files for m in members):
        add("Equal-weight ensemble (for comparison)", "+".join(members), np.mean([mo[m] for m in members], axis=0))
    add("Final ensemble (Stage 4b)", "+".join(members), d["morph"], d["merger"])
    T = load_json(ctx.require("4c", "temperature.json"))
    add("Final ensemble + calibration (final model)", "+".join(members),
        softmax_np(np.log(np.clip(d["morph"], 1e-12, 1)) / T["T_morph"]), d["merger"])
    if pca_k:
        P, n, nf = cv_subset(ALL, k=0)
        add("Full fusion without CNN PCA (all 1,806 raw columns)", "resnet18+efficientnet_b0+physical", P,
            n_trees=n, n_features=nf)
    if conf is not None:
        P, n, nf = cv_subset(ALL, cf=None)
        add("Full fusion without label-confidence weights", "resnet18+efficientnet_b0+physical", P, n_trees=n,
            n_features=nf)
    if cfg.get("cnn_crop", False) or int(cfg.get("cnn_tta_views", 1)) > 1:
        ctx.log.info(f"[Stage 4d] CNN features of the whole 128 px cutout (first-run input) for {len(pool)} galaxies ...")
        X2 = X.copy()
        X2[:, :1792] = _features_at(ctx, pool, extract_cnn_features, cnn_crop=False, cnn_mask_neighbours=False,
                                    cnn_clean_cosmic_rays=False, cnn_tta_views=1)
        P, n, nf = cv_subset(ALL, X2)
        add("Full fusion, whole 128 px cutout (no crop / mask / CR clean / rotations)",
            "resnet18+efficientnet_b0+physical", P, n_trees=n, n_features=nf)

    small = int(cfg.get("ablation_input_size") or 0)
    if small and small != int(meta.get("input_size", 224)):
        ctx.log.info(f"[Stage 4d] CNN features at {small} px for the {len(pool)} labelled galaxies ...")
        X2 = X.copy()
        X2[:, :1792] = _features_at(ctx, pool, extract_cnn_features, input_size=small)
        P, n, nf = cv_subset(ALL, X2)
        add(f"Full fusion, {small} px CNN input", "resnet18+efficientnet_b0+physical", P, n_trees=n, n_features=nf)

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
    images_only = [c for c in ("CNN features only (ResNet-18 + EfficientNet-B0)", "Fine-tuned CNNs only (average of both)",
                               "Fine-tuned ResNet-18 only", "Fine-tuned EfficientNet-B0 only") if c in f]
    delta = float(f["Final ensemble (Stage 4b)"] - max([f[c] for c in images_only] + [f["Physical features only (classical)"]]))
    tier = "Stretch" if delta >= 0.05 else "Target" if delta >= 0.03 else "Minimum" if delta >= 0 else "Below minimum"
    save_json({"fusion_delta_macro_f1": round(delta, 4), "tier": tier,
               "scored_on": "out-of-fold predictions, pooled train + dev galaxies, Stage 4b folds"}, out / "summary.json")
    _plot(tab[tab.task.str.startswith("morphology")], out)
    ctx.log.info("[Stage 4d]\n" + tab.drop(columns=["task"]).to_string(index=False))
    ctx.log.info(f"[Stage 4d] final ensemble minus the best single source (images only or physics only) = "
                 f"{delta:+.4f} -> {tier}")
    return {"fusion_delta": round(delta, 4), "tier": tier}


def _features_at(ctx, rows, extract, **override):
    """CNN features with other supervised settings (the config is changed for this call only)."""
    sup = ctx.cfg["supervised"]
    old = {k: sup.get(k) for k in override}
    sup.update(override)
    try:
        return extract(ctx, rows)
    finally:
        sup.update(old)


def _plot(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = t.sort_values("macro_f1")
    col = ["#eb6834" if "final model" in c else "#2E5496" if c.startswith(("Full fusion", "Final ensemble")) else
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
