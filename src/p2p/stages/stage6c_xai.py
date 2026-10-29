"""Stage 6c - Explainable AI: which evidence drives each prediction (exact TreeSHAP of the fusion XGBoost).

Method  : XGBoost's built-in pred_contribs gives the exact SHAP value of every one of the 1,806 inputs for every
          test galaxy and class. They are summarised (mean |SHAP| over the test galaxies)
            * per ENSEMBLE block - how much of each class decision rests on ResNet-18 features, EfficientNet-B0
              features and the physical features;
            * per physical feature - which measured quantity (Gini, colour, ellipticity, ...) matters per class;
          and the most confident correctly classified test galaxy of each class is shown as a colour cutout.
Output  : shap_blocks.csv, shap_physical.csv, shap_summary.png, examples.png, summary.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, StageSkipped, save_json
from ..features import load_cutouts, supervised_splits
from ..inference import BLOCKS, block_slices, load_fusion, predict_galaxies
from ..nets import MORPH_CLASSES, N_CLASSES


def run(ctx: Ctx):
    from ..sealing import require_unsealed
    require_unsealed(ctx, "6c")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ctx.stage_dir("6c")
    if not ctx.path("6a", "test_predictions.csv").exists():
        raise StageSkipped("Stage 6a has no test predictions (no labelled galaxies to evaluate)")
    pred = pd.read_csv(ctx.require("6a", "test_predictions.csv"))
    test = supervised_splits(ctx)["test"].set_index("object_id").loc[pred.object_id].reset_index()
    models = load_fusion(ctx)
    physical = models["meta"]["physical_columns"]
    X = predict_galaxies(ctx, test, models)["X"]
    c = models["morph"].contribs(X)                      # (n, classes present in training, 1806)
    cls = [MORPH_CLASSES[k] for k in models["morph"].classes_]
    A = np.abs(c).mean(0)                                # (classes, features)
    sl = block_slices(len(physical))
    blocks = pd.DataFrame({b: A[:, sl[b]].sum(1) for b in BLOCKS}, index=cls)
    shares = blocks.div(blocks.sum(1), axis=0)
    phys = pd.DataFrame(A[:, sl["physical"]], index=cls, columns=physical)
    shares.to_csv(out / "shap_blocks.csv")
    phys.to_csv(out / "shap_physical.csv")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    left = np.zeros(len(cls))
    for b, col in zip(BLOCKS, ("#2a78d6", "#1baf7a", "#eb6834")):
        axs[0].barh(cls, shares[b], left=left, color=col, label=b)
        left += shares[b].to_numpy()
    axs[0].set_xlabel("share of total |SHAP| per class")
    axs[0].set_title("Which ENSEMBLE block drives each class")
    axs[0].legend(fontsize=8, loc="lower right")
    phys.T.plot.barh(ax=axs[1], width=0.8)
    axs[1].set_xlabel("mean |SHAP| (log-odds)")
    axs[1].set_title("Physical features per class")
    fig.tight_layout()
    fig.savefig(out / "shap_summary.png", dpi=110)
    plt.close(fig)

    # most confident correct test galaxy per class, as a colour cutout (R, G, B = reddest .. bluest band)
    P = pred[[f"p_{c_}" for c_ in MORPH_CLASSES]].to_numpy()
    picks = [ok[np.argmax(P[ok, k])] for k in range(N_CLASSES)
             if (ok := np.where((pred.y_pred == k) & (pred.y_true == k))[0]).size]
    if picks:
        cut = load_cutouts(ctx)
        fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 2.9), squeeze=False)
        for ax, i in zip(axes[0], picks):
            img = np.asarray(cut[int(test.cutout_index[i])], np.float32)[::-1].transpose(1, 2, 0)
            lo, hi = np.percentile(img, [1, 99.5])
            v = np.clip((img - lo) / max(hi - lo, 1e-8), 0, 1)
            ax.imshow(np.arcsinh(v / 0.1) / np.arcsinh(10), origin="lower")
            k = int(pred.y_pred[i])
            ax.set_title(f"{MORPH_CLASSES[k]}\np = {P[i, k]:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle("Stage 6c: most confident correct test galaxy per class")
        fig.tight_layout()
        fig.savefig(out / "examples.png", dpi=100)
        plt.close(fig)
    save_json({"method": "XGBoost TreeSHAP (pred_contribs), fusion model", "n_test": int(len(test)),
               "block_share_by_class": shares.round(4).to_dict(orient="index"),
               "example_classes": [MORPH_CLASSES[int(pred.y_pred[i])] for i in picks]}, out / "summary.json")
    ctx.log.info("[Stage 6c] share of |SHAP| by ENSEMBLE block:\n" + shares.round(3).to_string()
                 + "\n[Stage 6c] physical features (mean |SHAP|):\n" + phys.round(3).to_string())
    return {"classes": len(cls)}
