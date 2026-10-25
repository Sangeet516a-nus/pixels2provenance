"""Stage 6c - Explainable AI: Grad-CAM and SHAP.

Purpose : show what drives each prediction.
Method  : Grad-CAM on the last convolutional block (layer4) of the dual-branch
          ResNet-18, for the most confident correct test galaxy of each class;
          exact TreeSHAP attributions from XGBoost's built-in pred_contribs
          (mean |SHAP| per class and feature).
Output  : gradcam_examples.png, shap_summary.csv, shap_summary.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..context import Ctx, save_json
from ..features import TabularPrep, load_cutouts, make_tensors, supervised_splits
from ..inference import load_models
from ..nets import MORPH_CLASSES, N_CLASSES


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.h = [layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o)),
                  layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))]

    def __call__(self, x_img, x_tab, device):
        self.model.to(device).eval()
        x_img, x_tab = x_img.to(device), x_tab.to(device)
        with torch.enable_grad():
            logits = self.model(x_img, x_tab)["morph"]
            k = int(logits.argmax(1))
            self.model.zero_grad()
            logits[0, k].backward()
        cam = F.relu((self.grads.mean(dim=(2, 3), keepdim=True) * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x_img.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return cam / (cam.max() + 1e-12), k

    def remove(self):
        for h in self.h:
            h.remove()


def run(ctx: Ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ctx.stage_dir("6c")
    models = load_models(ctx)
    pred = pd.read_csv(ctx.require("6a", "test_predictions.csv"))
    test_df = supervised_splits(ctx)["test"]
    data = make_tensors(test_df, load_cutouts(ctx), TabularPrep.load(ctx.require("4a", "tab_prep.joblib")))
    P = pred[[f"p_{c}" for c in MORPH_CLASSES]].to_numpy()
    picks = []
    for k in range(N_CLASSES):
        ok = np.where((pred.y_pred == k) & (pred.y_true == k))[0]
        if ok.size:
            picks.append(ok[np.argmax(P[ok, k])])
    cam = GradCAM(models["dual"], models["dual"].backbone.layer4)
    if picks:
        fig, axes = plt.subplots(2, len(picks), figsize=(2.4 * len(picks), 5), squeeze=False)
        for j, i in enumerate(picks):
            heat, k = cam(data["img"][i:i + 1], data["tab"][i:i + 1], ctx.device())
            base = data["img"][i].numpy().mean(0)
            axes[0, j].imshow(base, origin="lower", cmap="gray")
            axes[0, j].set_title(MORPH_CLASSES[k], fontsize=9)
            axes[1, j].imshow(base, origin="lower", cmap="gray")
            axes[1, j].imshow(heat, origin="lower", cmap="jet", alpha=0.45)
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle("Stage 6c: Grad-CAM (dual-branch ResNet-18, layer4)")
        fig.tight_layout()
        fig.savefig(out / "gradcam_examples.png", dpi=100)
        plt.close(fig)
    cam.remove()

    tab = models["tabular"]
    names = models["prep"].kept
    X = data["tab"].numpy()
    if tab.kind == "xgboost":
        c = tab.contribs(X)                   # (n, classes, features), computed in XGBoost's own process
        idx = [MORPH_CLASSES[k] for k in tab.classes_] if c.shape[1] == len(tab.classes_) else ["all classes"]
        shap = pd.DataFrame(np.abs(c).mean(0), index=idx, columns=names)
        method = "XGBoost TreeSHAP (pred_contribs)"
    elif tab.kind == "sklearn-hgb":
        from sklearn.inspection import permutation_importance
        remap = {c: i for i, c in enumerate(tab.classes_)}
        yt = test_df.gz_label_id.map(remap)
        ok = yt.notna().to_numpy()
        pi = permutation_importance(tab.model, X[ok], yt[ok].astype(int), n_repeats=10, random_state=ctx.seed)
        shap = pd.DataFrame([pi.importances_mean], index=["all classes"], columns=names)
        method = "permutation importance"
    else:                                     # a constant member has nothing to explain
        shap = pd.DataFrame([np.zeros(len(names))], index=["all classes"], columns=names)
        method = "none (tabular member is constant: one class in the training data)"
    shap.to_csv(out / "shap_summary.csv")
    ax = shap.T.plot.barh(figsize=(6.5, 0.35 * len(names) + 1.5), width=0.8)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("Stage 6c: tabular attributions per class")
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=110)
    plt.close()
    save_json({"gradcam_classes": [MORPH_CLASSES[int(pred.y_pred[i])] for i in picks], "attribution": method},
              out / "summary.json")
    ctx.log.info(f"[Stage 6c] Grad-CAM for {len(picks)} classes; {method}:\n" + shap.round(3).to_string())
    return {"attribution": method}
