"""Stage 6b - Uncertainty audit.

Purpose : does the model hesitate where humans disagree?
Method  : normalised predictive entropy of each test galaxy is compared with
          human disagreement (1 - the weakest GZH vote fraction along the path
          to its label): Spearman correlation, plus a one-sided Mann-Whitney U
          test that the Ambiguous Boundary set (agreement < 0.5) has higher
          entropy than confidently labelled galaxies.
Output  : uncertainty_audit.json, uncertainty_audit.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from ..context import Ctx, save_json
from ..features import supervised_splits
from ..nets import MORPH_CLASSES
from ..utils import as_bool, predictive_entropy


def run(ctx: Ctx):
    out = ctx.stage_dir("6b")
    pred = pd.read_csv(ctx.require("6a", "test_predictions.csv"))
    test = supervised_splits(ctx)["test"][["object_id", "human_disagreement", "in_boundary_set"]]
    d = pred.merge(test, on="object_id")
    P = d[[f"p_{c}" for c in MORPH_CLASSES]].to_numpy()
    e = predictive_entropy(P)
    h = d.human_disagreement.to_numpy(float)
    b = as_bool(d.in_boundary_set).to_numpy()
    rho, p = spearmanr(e, h) if len(d) > 2 else (np.nan, np.nan)
    acc = (d.y_true == d.y_pred).to_numpy()
    res = {"n_test": int(len(d)), "spearman_entropy_vs_disagreement": round(float(rho), 4),
           "spearman_p": float(p), "n_boundary": int(b.sum()), "n_confident": int((~b).sum()),
           "mean_entropy_boundary": float(e[b].mean()) if b.any() else None,
           "mean_entropy_confident": float(e[~b].mean()) if (~b).any() else None,
           "accuracy_boundary": float(acc[b].mean()) if b.any() else None,
           "accuracy_confident": float(acc[~b].mean()) if (~b).any() else None}
    if b.sum() >= 3 and (~b).sum() >= 3:
        res["mannwhitney_p_boundary_gt_confident"] = float(mannwhitneyu(e[b], e[~b], alternative="greater").pvalue)
    else:
        res["note"] = "fewer than 3 boundary-set galaxies in the test split - the U-test is not run"
    save_json(res, out / "uncertainty_audit.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.scatter(h[~b], e[~b], s=12, alpha=0.6, label="confident (humans agree)")
    ax.scatter(h[b], e[b], s=18, marker="x", label="Ambiguous Boundary set")
    ax.set_xlabel("human disagreement (1 - GZH path agreement)")
    ax.set_ylabel("model predictive entropy")
    ax.set_title(f"Stage 6b: Spearman rho = {rho:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "uncertainty_audit.png", dpi=110)
    plt.close(fig)
    ctx.log.info(f"[Stage 6b] {res}")
    return res
