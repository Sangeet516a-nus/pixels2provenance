"""Stage 4d - Controlled ablation.

Purpose : measure the value the hybrid design adds (Objective 3).
Method  : CNN-only and classical-only models are the SAME DualBranchNet with
          one branch removed, trained with the same recipe and epoch budget
          (the tiny tabular-only net gets more epochs: that can only strengthen
          the baseline). All rows are scored on DEV - the test split is used
          once, in Stage 6a. The segregation rows compare the Stage-2c baseline
          with the learned CAE + clustering (Stage 2d).
Output  : ablation_table.csv, ablation.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_json, save_json
from ..features import TabularPrep, load_cutouts, make_tensors, supervised_splits
from ..inference import calibrated_predict, load_models
from ..nets import DualBranchNet, fit_classifier, morph_metrics, predict_model
from ..utils import expected_calibration_error


def run(ctx: Ctx):
    cfg = ctx.cfg["supervised"]
    out = ctx.stage_dir("4d")
    splits = supervised_splits(ctx)
    cut = load_cutouts(ctx)
    prep = TabularPrep.load(ctx.require("4a", "tab_prep.joblib"))
    data = {s: make_tensors(v, cut, prep) for s, v in splits.items() if s != "test"}
    meta = load_json(ctx.require("4a", "model_meta.json"))
    common = dict(batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"], gamma=cfg["focal_gamma"],
                  merger_weight=cfg["merger_weight"], seed=ctx.seed, log=ctx.log.info)
    img_only = fit_classifier(DualBranchNet(meta["n_tab"], in_ch=meta["in_ch"], mode="image", pretrained=cfg["pretrained"]),
                              data["train"], data["dev"], ctx.device(), epochs=cfg["epochs"], lr=cfg["lr"],
                              tag="cnn-only", **common)
    tab_only = fit_classifier(DualBranchNet(meta["n_tab"], in_ch=meta["in_ch"], mode="tabular"),
                              data["train"], data["dev"], ctx.device(), epochs=cfg["tabular_only_epochs"], lr=3e-3,
                              tag="classical-only", **common)
    models = load_models(ctx)
    dev = data["dev"]
    y, ym = dev["y"].numpy(), dev["ym"].numpy()
    rows = []

    def add(name, rep, P, pm=None):
        m = morph_metrics(y, P, ym, pm)
        rows.append({"task": "morphology (dev)", "configuration": name, "representation": rep,
                     "macro_f1": m["macro_f1"], "balanced_acc": m["balanced_acc"],
                     "merger_pr_auc": m.get("merger_pr_auc", np.nan),
                     "ece": round(expected_calibration_error(P, y), 4)})

    for name, rep, mdl in (("Classical-only (Dense branch)", "tabular", tab_only),
                           ("CNN-only (ResNet-18 branch)", "image", img_only),
                           ("Hybrid fusion (ResNet-18 + Dense)", "image+tabular", models["dual"])):
        p = predict_model(mdl, dev, ctx.device())
        add(name, rep, p["morph"], p.get("merger"))
    add("XGBoost member (tabular)", "tabular", models["tabular"].predict_proba(dev["tab"].numpy()))
    add("EfficientNet member (image)", "image", predict_model(models["effnet"], dev, ctx.device())["morph"])
    cal = calibrated_predict(ctx, dev, models)
    add("Full ensemble (soft vote + T-scaling)", "ensemble", cal["morph"], cal["merger"])
    audit = load_json(ctx.require("2d", "semantic_audit.json"))
    for name, key in (("Baseline segregator (Stage 2c, no Gaia leg)", "baseline_star_vs_gaia"),
                      ("CAE + GMM/HDBSCAN (Stage 2d)", "learned_star_vs_gaia")):
        m = audit[key]
        rows.append({"task": "star segregation vs Gaia", "configuration": name, "star_precision": m["precision"],
                     "star_recall": m["recall"], "star_f1": m["f1"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ablation_table.csv", index=False)
    f = tab.set_index("configuration").macro_f1
    delta = float(f["Hybrid fusion (ResNet-18 + Dense)"] - max(f["Classical-only (Dense branch)"],
                                                              f["CNN-only (ResNet-18 branch)"]))
    tier = "Stretch" if delta >= 0.05 else "Target" if delta >= 0.03 else "Minimum" if delta >= 0 else "Below minimum"
    save_json({"hybrid_delta_macro_f1": round(delta, 4), "tier": tier}, out / "summary.json")
    _plot(tab[tab.task == "morphology (dev)"], out)
    ctx.log.info("[Stage 4d]\n" + tab.to_string(index=False))
    ctx.log.info(f"[Stage 4d] hybrid minus best single branch = {delta:+.4f} -> {tier}")
    return {"hybrid_delta": round(delta, 4), "tier": tier}


def _plot(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = t.sort_values("macro_f1")
    col = ["#2E5496" if "Hybrid" in c else "#9AA3B5" if "branch" in c else "#D3D9E5" for c in t.configuration]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(t.configuration, t.macro_f1, color=col)
    for i, v in enumerate(t.macro_f1):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("dev Macro-F1")
    ax.set_title("Stage 4d: controlled ablation (blue = hybrid, dark grey = its single branches)")
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=110)
    plt.close(fig)
