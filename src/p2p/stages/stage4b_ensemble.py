"""Stage 4b - Ensemble consensus.

Purpose : combine models that make different errors.
Method  : a standalone EfficientNet-B0 (pixels only) and an XGBoost model
          (Pass-2 table only; balanced sample weights, early stopping on dev)
          are trained and soft-voted with the dual-branch network
          (supervised.ensemble_weights). XGBoost runs in its own process (see
          p2p/xgb_worker.py: on macOS it can crash next to PyTorch). The merger probability comes from the
          dual-branch model alone - the other members have no merger head.
Output  : efficientnet.pt, tabular_member.joblib, dev_metrics.json
"""
from __future__ import annotations

import joblib
import torch

from ..context import Ctx, save_json
from ..features import TabularPrep, load_cutouts, make_tensors, supervised_splits
from ..inference import TabularMember, ensemble_predict, load_models
from ..nets import EfficientNetMember, fit_classifier, morph_metrics


def run(ctx: Ctx):
    cfg = ctx.cfg["supervised"]
    out = ctx.stage_dir("4b")
    splits = supervised_splits(ctx)
    cut = load_cutouts(ctx)
    prep = TabularPrep.load(ctx.require("4a", "tab_prep.joblib"))
    data = {s: make_tensors(v, cut, prep) for s, v in splits.items() if s != "test"}

    eff = fit_classifier(EfficientNetMember(in_ch=cut.shape[1], pretrained=cfg["pretrained"]), data["train"],
                         data["dev"], ctx.device(), epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                         weight_decay=cfg["weight_decay"], gamma=cfg["focal_gamma"], seed=ctx.seed,
                         log=ctx.log.info, tag="efficientnet")
    torch.save(eff.state_dict(), out / "efficientnet.pt")
    tab = TabularMember(seed=ctx.seed, log=ctx.log.info).fit(data["train"]["tab"].numpy(), data["train"]["y"].numpy(),
                                           data["dev"]["tab"].numpy(), data["dev"]["y"].numpy())
    joblib.dump(tab, out / "tabular_member.joblib")

    models = load_models(ctx)
    ens = ensemble_predict(ctx, data["dev"], models)
    y, ym = data["dev"]["y"].numpy(), data["dev"]["ym"].numpy()
    res = {k: morph_metrics(y, P) for k, P in ens["members"].items()}
    res["ensemble"] = morph_metrics(y, ens["morph"], ym, ens["merger"])
    res["tabular_member_kind"] = tab.kind
    save_json(res, out / "dev_metrics.json")
    for k in ("dual", "effnet", "tabular", "ensemble"):
        ctx.log.info(f"[Stage 4b] {k:9s} dev Macro-F1 = {res[k]['macro_f1']:.3f}")
    return {"ensemble_dev_macro_f1": res["ensemble"]["macro_f1"]}
