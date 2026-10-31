"""Stage 6c - Explainable AI: which evidence drives each prediction (exact TreeSHAP of the XGBoost models).

Method  : XGBoost's built-in pred_contribs gives the exact SHAP value of every input for every test galaxy and class.
          They are summarised (mean |SHAP| over the test galaxies)
            * full-fusion XGBoost (frozen CNN features + physical, Stage 4b): per feature BLOCK - how much of each
              class decision rests on ResNet-18 features, EfficientNet-B0 features and the physical features
              (which evidence source the data supports; this model is a diagnostic unless it is an ensemble member);
            * per physical feature - which measured quantity (Gini, colour, ellipticity, ...) matters per class,
              for the fusion XGBoost AND for the physical XGBoost member of the final ensemble.
          TreeSHAP covers tree models only: the fine-tuned CNN members are explained by Grad-CAM when
          evaluation.gradcam is true (their share of the final decision is visible in the Stage 4d member rows).
          The most confident correctly classified test galaxy of each class is shown as a colour cutout.
Output  : shap_blocks.csv, shap_physical.csv, shap_physical_member.csv, shap_summary.png, examples.png, summary.json,
          gradcam_<network>.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, StageSkipped, save_json
from ..features import load_cutouts, supervised_splits
from ..inference import BLOCKS, block_slices, load_fusion, predict_galaxies
from ..nets import MORPH_CLASSES, N_CLASSES


def run(ctx: Ctx):
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
    c = models["morph"].contribs(X)                      # (n, classes present in training, model inputs)
    cls = [MORPH_CLASSES[k] for k in models["morph"].classes_]
    A = np.abs(c).mean(0)                                # (classes, features)
    mb = models["meta"].get("blocks")                    # columns of the model input (after the CNN PCA)
    sl = {b: slice(a, z) for b, (a, z) in mb.items()} if mb else block_slices(len(physical))
    blocks = pd.DataFrame({b: A[:, sl[b]].sum(1) for b in BLOCKS}, index=cls)
    shares = blocks.div(blocks.sum(1), axis=0)
    phys = pd.DataFrame(A[:, sl["physical"]], index=cls, columns=physical)
    shares.to_csv(out / "shap_blocks.csv")
    phys.to_csv(out / "shap_physical.csv")
    member = None
    if models.get("physical") is not None:            # physical XGBoost member of the final ensemble
        from ..inference import physical_matrix
        cm = models["physical"].contribs(physical_matrix(test, physical))
        member = pd.DataFrame(np.abs(cm).mean(0)[:, :len(physical)],
                              index=[MORPH_CLASSES[k] for k in models["physical"].classes_], columns=physical)
        member.to_csv(out / "shap_physical_member.csv")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    left = np.zeros(len(cls))
    for b, col in zip(BLOCKS, ("#2a78d6", "#1baf7a", "#eb6834")):
        axs[0].barh(cls, shares[b], left=left, color=col, label=b)
        left += shares[b].to_numpy()
    axs[0].set_xlabel("share of total |SHAP| per class")
    axs[0].set_title("Which ENSEMBLE block drives each class")
    axs[0].legend(fontsize=8, loc="lower right")
    (member if member is not None else phys).T.plot.barh(ax=axs[1], width=0.8)
    axs[1].set_xlabel("mean |SHAP| (log-odds)")
    axs[1].set_title("Physical features per class" + (" (physical XGBoost member)" if member is not None
                                                      else " (fusion XGBoost)"))
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
    cam = _gradcam(ctx, test, pred, models["members"], out) if ctx.cfg.get("evaluation", {}).get("gradcam") else None
    save_json({"method": "XGBoost TreeSHAP (pred_contribs): fusion XGBoost blocks + physical XGBoost member",
               "gradcam_centre_share": cam,
               "ensemble_members": models["members"], "n_test": int(len(test)),
               "block_share_by_class": shares.round(4).to_dict(orient="index"),
               "example_classes": [MORPH_CLASSES[int(pred.y_pred[i])] for i in picks]}, out / "summary.json")
    ctx.log.info("[Stage 6c] share of |SHAP| by ENSEMBLE block:\n" + shares.round(3).to_string()
                 + "\n[Stage 6c] physical features (mean |SHAP|, fusion XGBoost):\n" + phys.round(3).to_string()
                 + ("" if member is None else "\n[Stage 6c] physical features (mean |SHAP|, physical XGBoost "
                    "member):\n" + member.round(3).to_string()))
    return {"classes": len(cls)}


def _gradcam(ctx, test, pred, members, out):
    """Grad-CAM of each fine-tuned CNN member (evaluation.gradcam): the most confident correct test galaxies per class,
    stamp and heat map side by side. Returns, per network, the mean share of the heat in the central half of the stamp
    (low = the network looks at the surroundings, not the galaxy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .. import finetune
    from ..features import IMAGENET_MEAN, IMAGENET_STD
    n = int(ctx.cfg["evaluation"].get("gradcam_per_class", 4))
    P = pred[[f"p_{c_}" for c_ in MORPH_CLASSES]].to_numpy()
    picks = [i for k in range(N_CLASSES) for i in sorted(np.where((pred.y_pred == k) & (pred.y_true == k))[0],
                                                         key=lambda j: -P[j, k])[:n]]
    if not picks:
        return None
    rows = test.iloc[picks]
    classes = pred.y_pred.to_numpy(int)[picks]
    result = {}
    for kind, member in finetune.MEMBER.items():
        if member not in members:
            continue
        model, cfg = finetune.load(ctx.require("4b", f"finetuned_{kind}.pt"), ctx.device())
        X = finetune.stamps(ctx, rows, cfg["input_size"])
        cams = finetune.grad_cam(model, X, classes, ctx.device())
        s = cams.shape[-1]
        centre = np.zeros((s, s), bool)
        centre[s // 4:3 * s // 4, s // 4:3 * s // 4] = True
        result[kind] = round(float((cams[:, centre].sum(1) / np.maximum(cams.sum((1, 2)), 1e-8)).mean()), 3)
        rgb = np.clip(X.numpy() * IMAGENET_STD + IMAGENET_MEAN, 0, 1).transpose(0, 2, 3, 1)
        ncol = 2 * max(n, 1)
        nrow = int(np.ceil(len(picks) * 2 / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.75 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for j in range(len(picks)):
            a, b = axes.flat[2 * j], axes.flat[2 * j + 1]
            a.imshow(rgb[j], origin="lower")
            a.set_title(MORPH_CLASSES[classes[j]], fontsize=7)
            b.imshow(rgb[j].mean(2), origin="lower", cmap="gray")
            b.imshow(cams[j], origin="lower", cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            b.set_title("Grad-CAM", fontsize=7)
        fig.suptitle(f"Stage 6c: Grad-CAM, fine-tuned {kind} (most confident correct test galaxies)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"gradcam_{kind}.png", dpi=110, facecolor="white")
        plt.close(fig)
    ctx.log.info(f"[Stage 6c] Grad-CAM: mean share of the heat in the central half of the stamp {result} "
                 "(gradcam_<network>.png)")
    return result
