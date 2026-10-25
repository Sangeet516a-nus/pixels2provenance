"""Stage 4a - Dual-branch multi-task training.

Purpose : the core supervised model of the proposal.
Method  : ResNet-18 image branch (512-d, one input channel per band) and a
          Dense(n_features -> 64) projection of the Pass-2 table, each
          LayerNorm-ed so the larger image vector cannot drown the tabular one,
          concatenated into Head A (5-class softmax) and Head B (merger sigmoid).
          Loss = class-weighted focal loss (gamma 2) + 0.5 x positive-weighted BCE.
          AdamW + cosine schedule, 90-degree rotations and flips as label-safe
          augmentation; fixed epoch budget, best dev Macro-F1 checkpoint kept.
          The tabular pre-processing (impute, signed log, RobustScaler) is
          fitted on the train split only.
Output  : dual_branch.pt, tab_prep.joblib, model_meta.json, dataset_index.csv, dev_metrics.json
"""
from __future__ import annotations

import pandas as pd
import torch

from ..context import Ctx, save_json
from ..features import TabularPrep, load_cutouts, make_tensors, supervised_splits
from ..nets import MORPH_CLASSES, DualBranchNet, fit_classifier, morph_metrics, predict_model


def run(ctx: Ctx):
    cfg = ctx.cfg["supervised"]
    out = ctx.stage_dir("4a")
    splits = supervised_splits(ctx)
    cut = load_cutouts(ctx)
    prep = TabularPrep(ctx.cfg["representation"]["tabular_features"]).fit(splits["train"])
    prep.save(out / "tab_prep.joblib")
    data = {s: make_tensors(v, cut, prep) for s, v in splits.items()}
    pd.concat([v.assign(split=s)[["object_id", "split", "gz_label", "gz_label_id", "gz_merger_flag"]]
               for s, v in splits.items()]).to_csv(out / "dataset_index.csv", index=False)
    n_tab, in_ch = data["train"]["tab"].shape[1], cut.shape[1]
    save_json({"n_tab": n_tab, "in_ch": in_ch, "tab_features": prep.kept, "classes": list(MORPH_CLASSES),
               "sizes": {s: len(v) for s, v in splits.items()}}, out / "model_meta.json")
    allrows = pd.concat([v.assign(split=s) for s, v in splits.items()], ignore_index=True)
    counts = pd.crosstab(allrows.split, allrows.gz_label)
    ctx.log.info(f"[Stage 4a] device {ctx.device()} | tabular features {prep.kept}\n" + counts.to_string())

    model = fit_classifier(DualBranchNet(n_tab, in_ch=in_ch, mode="hybrid", pretrained=cfg["pretrained"]),
                           data["train"], data["dev"], ctx.device(), epochs=cfg["epochs"],
                           batch_size=cfg["batch_size"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                           gamma=cfg["focal_gamma"], merger_weight=cfg["merger_weight"], seed=ctx.seed,
                           log=ctx.log.info, tag="dual-branch")
    torch.save(model.state_dict(), out / "dual_branch.pt")
    p = predict_model(model, data["dev"], ctx.device())
    m = morph_metrics(data["dev"]["y"].numpy(), p["morph"], data["dev"]["ym"].numpy(), p["merger"])
    save_json(m, out / "dev_metrics.json")
    ctx.log.info(f"[Stage 4a] dual-branch (dev): {m}")
    return m
