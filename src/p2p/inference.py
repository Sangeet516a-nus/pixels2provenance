"""The supervised model of the pipeline (feature-level fusion) and its frozen, calibrated inference path.

    galaxy cutout (128 x 128, 3 bands)  -> ImageNet input (R, G, B = F775W, F606W, F435W; asinh; 224 x 224)
        -> ResNet-18 (ImageNet, frozen)        -> 512 features      \
        -> EfficientNet-B0 (ImageNet, frozen)  -> 1,280 features     >  ENSEMBLE (fusion): one 1,806-feature vector,
    + 14 physical features (Pass 2)            -> 14 features       /   block weights w1, w2, w3
        -> XGBoost (5 morphology classes)  +  XGBoost (merger yes / no)  -> calibration (Stage 4c)

Used by Stages 4a (feature extraction), 4b (training), 4c, 4d, 6a, 6c, 7b and 8.
"""
from __future__ import annotations

import time

import joblib
import numpy as np

from .context import Ctx, load_json
from .features import imagenet_input, load_cutouts, physical_matrix
from .nets import N_CLASSES, logit, softmax_np

CNN_BLOCKS = (("resnet18", 512), ("efficientnet_b0", 1280))
BLOCKS = ("resnet18", "efficientnet_b0", "physical")


# ================================================================ Stage 4a: frozen CNN features
def build_extractors(pretrained=True):
    """The two frozen ImageNet networks (no training: BatchNorm uses its ImageNet statistics, Dropout is off)."""
    from .nets import EfficientNetB0, ResNet18, frozen
    return {"resnet18": frozen(ResNet18(in_ch=3, pretrained=pretrained)),
            "efficientnet_b0": frozen(EfficientNetB0(pretrained=pretrained))}


def extract_cnn_features(ctx: Ctx, cutout_index, nets=None, log=None):
    """(n,) cutout rows of Stage 1d's cutouts.npy -> (n, 1792) float32: [ResNet-18 512 | EfficientNet-B0 1280]."""
    import torch
    cfg = ctx.cfg["supervised"]
    size, bs = int(cfg.get("input_size", 224)), int(cfg.get("feature_batch_size", 64))
    nets = nets or build_extractors(bool(cfg.get("pretrained", True)))
    dev = ctx.device()
    for m in nets.values():
        m.to(dev)
    cut = load_cutouts(ctx)
    idx = np.asarray(cutout_index, int)
    out = np.zeros((len(idx), sum(n for _, n in CNN_BLOCKS)), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            rows = idx[s:s + bs]
            order = np.argsort(rows)                                   # sorted reads from the memory map
            x = imagenet_input(cut[rows[order]], size).to(dev)
            f = torch.cat([nets[k](x) for k, _ in CNN_BLOCKS], 1).cpu().numpy()
            out[s + order] = f
            if log and (s // bs) % 50 == 0:
                log(f"  [features] {min(s + bs, len(idx)):,} / {len(idx):,} cutouts ({time.time() - t0:.0f} s)")
    return out


def feature_names(physical):
    return ([f"resnet18_{i}" for i in range(512)] + [f"efficientnet_b0_{i}" for i in range(1280)] + list(physical))


def block_slices(n_physical):
    return {"resnet18": slice(0, 512), "efficientnet_b0": slice(512, 1792),
            "physical": slice(1792, 1792 + n_physical)}


def block_feature_weights(weights, n_physical):
    """ENSEMBLE weights w1, w2, w3 (sum 1) -> one sampling weight per feature: w_block / features in the block, so a
    block's share of the features each tree may use is w_block, whatever its size. (Multiplying the features
    themselves by w would do nothing: a tree split is a threshold, unchanged by scaling a feature.)"""
    sl = block_slices(n_physical)
    fw = np.zeros(1792 + n_physical)
    for b, w in zip(BLOCKS, weights):
        n = sl[b].stop - sl[b].start
        fw[sl[b]] = max(float(w), 1e-6) / n
    return fw


def fusion_matrix(ctx: Ctx, df, physical):
    """The 1,806-feature vector of every row of df: CNN features from Stage 4a + the physical columns."""
    ids = np.load(ctx.require("4a", "object_id.npy"))
    F = np.load(ctx.require("4a", "cnn_features.npy"), mmap_mode="r")
    pos = {int(o): i for i, o in enumerate(ids)}
    rows = np.array([pos.get(int(o), -1) for o in df["object_id"]])
    cnn = np.zeros((len(df), F.shape[1]), np.float32)
    have = rows >= 0
    cnn[have] = F[rows[have]]
    if (~have).any():                                  # a galaxy Stage 4a did not see: extract it now
        cnn[~have] = extract_cnn_features(ctx, df["cutout_index"].to_numpy()[~have])
    return np.hstack([cnn, physical_matrix(df, physical)])


# ================================================================ XGBoost members (Stage 4b)
class XGBMember:
    """One XGBoost classifier (5 morphology classes, or merger yes / no) with balanced sample weights.
    Runs in a separate process (p2p.xgb_worker): next to PyTorch on macOS XGBoost can crash Python.
    Classes missing from the training data are mapped back to all output columns."""

    def __init__(self, n_classes, seed=0, params=None, feature_weights=None):
        self.n_classes, self.seed, self.params = int(n_classes), seed, dict(params or {})
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights, float)
        self.classes_, self.raw, self.n_estimators = None, None, None

    @staticmethod
    def _prepare(y):
        classes = np.unique(y)
        remap = {c: i for i, c in enumerate(classes)}
        yy = np.array([remap[c] for c in y])
        counts = np.bincount(yy)
        w = (len(yy) / (len(counts) * np.maximum(counts, 1)))[yy]          # balanced sample weights
        return classes, remap, yy, w

    def cv_curve(self, X, y, Xva, yva, grid):
        """Held-out probabilities (len(grid), n_va, n_classes) after each tree count in grid + history."""
        from . import xgb_worker
        classes, remap, yy, w = self._prepare(y)
        yv = np.array([remap.get(c, -1) for c in yva])
        p, history = xgb_worker.run_isolated(xgb_worker.fit_curve, np.asarray(X, np.float32), yy, w,
                                             np.asarray(Xva, np.float32), yv, len(classes), list(grid), self.seed,
                                             self.params, self.feature_weights)
        out = np.zeros((len(grid), len(Xva), self.n_classes))
        out[:, :, classes] = p
        return out / out.sum(2, keepdims=True), history

    def fit(self, X, y, n_estimators):
        from . import xgb_worker
        self.classes_, _, yy, w = self._prepare(y)
        self.n_estimators = int(n_estimators)
        self.raw = xgb_worker.run_isolated(xgb_worker.fit_fixed, np.asarray(X, np.float32), yy, w, self.n_estimators,
                                           len(self.classes_), self.seed, self.params, self.feature_weights)
        return self

    def predict_proba(self, X):
        from . import xgb_worker
        out = np.zeros((len(X), self.n_classes))
        out[:, self.classes_] = xgb_worker.run_isolated(xgb_worker.predict, self.raw, np.asarray(X, np.float32))
        return out / out.sum(1, keepdims=True)

    def contribs(self, X):
        """TreeSHAP values (n, outputs, features)."""
        from . import xgb_worker
        return xgb_worker.run_isolated(xgb_worker.contribs, self.raw, np.asarray(X, np.float32))

    def summary(self, names):
        from . import xgb_worker
        return xgb_worker.run_isolated(xgb_worker.summary, self.raw, list(names), self.params)


def xgb_cv(X, y, folds, grid, n_classes, seed, params, feature_weights=None):
    """5-fold CV of one XGBoost configuration: OOF probabilities after every tree count in grid
    (len(grid), n, n_classes) and the per-fold histories."""
    oof = np.zeros((len(grid), len(y), n_classes))
    hist = []
    for tr, va in folds:
        m = XGBMember(n_classes, seed, params, feature_weights)
        oof[:, va], h = m.cv_curve(X[tr], y[tr], X[va], y[va], grid)
        hist.append(h)
    return oof, hist


# ================================================================ frozen inference path
def load_fusion(ctx: Ctx):
    """The trained Stage 4b models + Stage 4c calibration."""
    m = {"meta": load_json(ctx.require("4b", "fusion_meta.json")),
         "morph": joblib.load(ctx.require("4b", "xgb_morphology.joblib")),
         "merger": joblib.load(ctx.require("4b", "xgb_merger.joblib")) if ctx.path("4b", "xgb_merger.joblib").exists()
         else None}
    p = ctx.path("4c", "temperature.json")
    m["calibration"] = load_json(p) if p.exists() else None
    return m


def predict_galaxies(ctx: Ctx, df, models=None, calibrated=True):
    """Class probabilities (n, 5) and merger probability (n,) of the galaxies in df (rows of galaxy_table)."""
    models = models or load_fusion(ctx)
    X = fusion_matrix(ctx, df, models["meta"]["physical_columns"])
    P = models["morph"].predict_proba(X)
    pm = models["merger"].predict_proba(X)[:, 1] if models["merger"] is not None else \
        np.full(len(df), models["meta"].get("merger_base_rate", 0.0))
    cal = models["calibration"]
    if calibrated and cal:
        P = softmax_np(np.log(np.clip(P, 1e-12, 1)) / cal["T_morph"])
        pm = 1 / (1 + np.exp(-(cal.get("merger_platt_a", 1.0) * logit(pm) + cal.get("merger_platt_b", 0.0))))
    return {"morph": P, "merger": pm, "X": X}


__all__ = ["BLOCKS", "CNN_BLOCKS", "N_CLASSES", "XGBMember", "block_feature_weights", "block_slices",
           "build_extractors", "extract_cnn_features", "feature_names", "fusion_matrix", "load_fusion",
           "predict_galaxies", "xgb_cv"]
