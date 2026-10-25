"""Loading the trained models and the frozen, calibrated inference path
(used by Stages 4b-4d, 6, 7 and 8)."""
from __future__ import annotations

import joblib
import numpy as np
import torch

from .context import Ctx, load_json
from .features import TabularPrep
from .nets import N_CLASSES, DualBranchNet, EfficientNetMember, logit, predict_model, softmax_np


class TabularMember:
    """XGBoost (or the sklearn fallback) on the Pass-2 table. XGBoost always runs in a separate
    process (p2p.xgb_worker): next to PyTorch on macOS it can crash Python with a segmentation
    fault. If that process fails, sklearn's HistGradientBoosting is used instead. Classes missing
    from the training data are handled by mapping back to all 5 columns."""

    def __init__(self, seed=0, log=print):
        self.seed = seed
        self.classes_ = None
        self.model = None          # sklearn model (fallback) - XGBoost lives in self.raw instead
        self.raw, self.best = None, None
        self.kind = None
        self._log = log

    def fit(self, X, y, Xdev=None, ydev=None):
        from . import xgb_worker
        self.classes_ = np.unique(y)
        remap = {c: i for i, c in enumerate(self.classes_)}
        yy = np.array([remap[c] for c in y])
        counts = np.bincount(yy)
        w = (len(yy) / (len(counts) * np.maximum(counts, 1)))[yy]
        if len(self.classes_) == 1:                        # nothing to learn: always that class
            self.kind = "constant"
            return self
        Xe = ye = None
        if Xdev is not None:
            keep = np.isin(ydev, self.classes_)
            if keep.any():
                Xe, ye = np.asarray(Xdev)[keep], np.array([remap[c] for c in np.asarray(ydev)[keep]])
        try:
            self.raw, self.best = xgb_worker.run_isolated(xgb_worker.fit, np.asarray(X, np.float32), yy, w, Xe, ye,
                                                          self.seed)
            self.kind = "xgboost"
        except Exception as e:                              # ImportError, or the child process crashed
            self._log(f"[TabularMember] XGBoost unavailable ({type(e).__name__}: {e}); using sklearn "
                      "HistGradientBoosting instead")
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                        class_weight="balanced", random_state=self.seed).fit(X, yy)
            self.kind = "sklearn-hgb"
        return self

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_log", None)                                 # the logger is not saved with the model
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._log = print

    def predict_proba(self, X):
        from . import xgb_worker
        X = np.asarray(X, np.float32)
        out = np.zeros((len(X), N_CLASSES))
        if self.kind == "constant":
            out[:, self.classes_[0]] = 1.0
            return out
        p = xgb_worker.run_isolated(xgb_worker.predict, self.raw, self.best, X) if self.kind == "xgboost" \
            else self.model.predict_proba(X)
        out[:, self.classes_] = p
        return out / out.sum(1, keepdims=True)

    def contribs(self, X):
        """TreeSHAP values (n, classes, features) - XGBoost only."""
        from . import xgb_worker
        return xgb_worker.run_isolated(xgb_worker.contribs, self.raw, self.best, np.asarray(X, np.float32))


def load_models(ctx: Ctx):
    meta = load_json(ctx.require("4a", "model_meta.json"))
    prep = TabularPrep.load(ctx.require("4a", "tab_prep.joblib"))
    dual = DualBranchNet(meta["n_tab"], in_ch=meta["in_ch"], mode="hybrid")
    dual.load_state_dict(torch.load(ctx.require("4a", "dual_branch.pt"), map_location="cpu"))
    m = {"meta": meta, "prep": prep, "dual": dual}
    if ctx.path("4b", "efficientnet.pt").exists():
        eff = EfficientNetMember(in_ch=meta["in_ch"])
        eff.load_state_dict(torch.load(ctx.path("4b", "efficientnet.pt"), map_location="cpu"))
        m["effnet"] = eff
        m["tabular"] = joblib.load(ctx.require("4b", "tabular_member.joblib"))
    return m


def ensemble_predict(ctx: Ctx, data, models):
    dev = ctx.device()
    d = predict_model(models["dual"], data, dev)
    out = {"members": {"dual": d["morph"]}, "merger": d["merger"]}
    if "effnet" in models:
        out["members"]["effnet"] = predict_model(models["effnet"], data, dev)["morph"]
        out["members"]["tabular"] = models["tabular"].predict_proba(data["tab"].numpy())
        w = np.asarray(ctx.cfg["supervised"]["ensemble_weights"], float)
        keys = ["dual", "effnet", "tabular"]
        P = sum(wi * out["members"][k] for wi, k in zip(w / w.sum(), keys))
    else:
        P = d["morph"]
    out["morph"] = P / P.sum(1, keepdims=True)
    return out


def calibrated_predict(ctx: Ctx, data, models):
    """The FROZEN inference path: soft vote -> temperature scaling (Stage 4c)."""
    out = ensemble_predict(ctx, data, models)
    p = ctx.path("4c", "temperature.json")
    if p.exists():
        T = load_json(p)
        out["morph"] = softmax_np(np.log(np.clip(out["morph"], 1e-12, 1)) / T["T_morph"])
        z = T.get("merger_platt_a", 1.0) * logit(out["merger"]) + T.get("merger_platt_b", 0.0)
        out["merger"] = 1 / (1 + np.exp(-z))
    return out
