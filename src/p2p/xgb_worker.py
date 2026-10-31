"""XGBoost in its own process.

On macOS, PyTorch and XGBoost each load their own OpenMP runtime (libomp). Two OpenMP runtimes in one Python
process can crash it with a segmentation fault, which no try/except can catch. So every XGBoost call - fit,
predict, TreeSHAP - runs in a short-lived child process that never imports torch. Only the trained booster
(raw JSON bytes) travels back to the main process.

This module must NOT import torch (directly or through p2p.nets / p2p.inference).
"""
from __future__ import annotations

import numpy as np

# Defaults of the "compile" step; config.yaml supervised.xgboost overrides any of them.
#   max_depth / min_child_weight : size of each tree (shallow trees: ~1,100 galaxies, ~1,800 features)
#   learning_rate                : shrinkage of every tree
#   subsample / colsample_bytree : random rows / features per tree (the tree-model equivalent of dropout);
#                                  with feature_weights the features are drawn block by block (ensemble weights)
#   reg_lambda / reg_alpha       : L2 / L1 penalty on the leaf values (the equivalent of L2 weight regularisation)
XGB_PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3, min_child_weight=2,
                  reg_lambda=1.0, reg_alpha=0.0, tree_method="hist", n_jobs=4)


def run_isolated(fn, *args):
    """Run fn(*args) in a FRESH Python process (`python -m p2p.xgb_worker`) that imports nothing but
    this module - multiprocessing's 'spawn' would re-import the main script, and with it PyTorch.
    A crash of the child (e.g. a segmentation fault) raises RuntimeError here instead of killing the run."""
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    pkg_parent = str(Path(__file__).resolve().parents[1])            # the folder that holds p2p/
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d, "in.pkl"), Path(d, "out.pkl")
        inp.write_bytes(pickle.dumps((fn.__name__, args)))
        env = dict(os.environ, PYTHONPATH=pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", ""))
        r = subprocess.run([sys.executable, "-m", "p2p.xgb_worker", str(inp), str(out)], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"XGBoost process failed (exit code {r.returncode}): {r.stderr.strip()[-400:]}")
        return pickle.loads(out.read_bytes())


def loaded_modules():
    """Diagnostics: the heavy libraries loaded in the worker process (torch must not be one)."""
    import sys
    return sorted({m.split(".")[0] for m in sys.modules} & {"torch", "torchvision", "xgboost"})


def build_xgb(params, seed, n_estimators, n_classes, feature_weights=None, metrics=None):
    """model_xgb = XGBClassifier(...) - the XGBoost equivalent of building + compiling a Keras model.
    Objective: softmax over the classes ('multi:softprob'), or logistic for two classes (merger yes / no)."""
    from xgboost import XGBClassifier
    kw = dict(XGB_PARAMS, **(params or {}), random_state=seed, n_estimators=int(n_estimators))
    kw["objective"] = "multi:softprob" if n_classes > 2 else "binary:logistic"
    if feature_weights is not None:
        kw["feature_weights"] = np.asarray(feature_weights, float)
    if metrics is not None:
        kw["eval_metric"] = metrics
    return XGBClassifier(**kw)


def fit_curve(X, y, w, Xva, yva, n_classes, grid, seed, params=None, feature_weights=None):
    """Cross-validation helper: fit max(grid) trees on (X, y) and return
      * the class probabilities of Xva after each tree count in grid -> (len(grid), n_va, n_classes)
      * the Keras-style history per boosting round: loss / accuracy on the training rows and val_loss /
        val_accuracy on the held-out fold (rows of a class the training fold lacks are not scored)."""
    import xgboost as xgb
    loss, err = ("mlogloss", "merror") if n_classes > 2 else ("logloss", "error")
    model_xgb = build_xgb(params, seed, max(grid), n_classes, feature_weights, metrics=[loss, err])
    keep = np.asarray(yva) >= 0
    eval_set = [(X, y)] + ([(np.asarray(Xva)[keep], np.asarray(yva)[keep])] if keep.any() else [])
    model_xgb.fit(X, y, sample_weight=w, eval_set=eval_set, verbose=False)
    res = model_xgb.evals_result()
    history = {"loss": res["validation_0"][loss], "accuracy": [1 - e for e in res["validation_0"][err]]}
    if "validation_1" in res:
        history.update(val_loss=res["validation_1"][loss], val_accuracy=[1 - e for e in res["validation_1"][err]])
    b = model_xgb.get_booster()
    d = xgb.DMatrix(np.asarray(Xva, np.float32))
    out = []
    for g in grid:
        p = np.asarray(b.predict(d, iteration_range=(0, int(g))))
        out.append(np.column_stack([1 - p, p]) if p.ndim == 1 else p)
    return np.stack(out), history


def fit_fixed(X, y, w, n_estimators, n_classes, seed, params=None, feature_weights=None):
    """Final model: the number of trees chosen by cross-validation, trained on all pooled galaxies."""
    model_xgb = build_xgb(params, seed, n_estimators, n_classes, feature_weights)
    model_xgb.fit(X, y, sample_weight=w)
    return bytes(model_xgb.get_booster().save_raw("json"))


def _booster(raw):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(bytearray(raw))
    return b, xgb


def predict(raw, X):
    b, xgb = _booster(raw)
    p = np.asarray(b.predict(xgb.DMatrix(np.asarray(X, np.float32))))
    return np.column_stack([1 - p, p]) if p.ndim == 1 else p          # binary -> two columns


def contribs(raw, X):
    """Exact TreeSHAP values, shape (n, classes, features) without the bias column (binary: classes = 1)."""
    b, xgb = _booster(raw)
    c = np.asarray(b.predict(xgb.DMatrix(np.asarray(X, np.float32)), pred_contribs=True))
    if c.ndim == 2:
        c = c[:, None, :]
    return c[:, :, :-1]


def summary(raw, feature_names, params=None):
    """model.summary() of a trained booster: trees, leaves, parameters and the gain importance of each input."""
    import json
    b, _ = _booster(raw)
    trees = b.get_dump(dump_format="json")
    leaves = [sum(1 for _ in _leaves(json.loads(t))) for t in trees]
    cfg = json.loads(b.save_config())["learner"]
    n_class = int(cfg["learner_model_param"]["num_class"]) or 1
    gain = b.get_score(importance_type="gain")
    imp = {feature_names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v for k, v in gain.items()}
    return {"n_trees": len(trees), "boosting_rounds": len(trees) // n_class, "outputs": n_class,
            "leaves_total": int(sum(leaves)), "leaves_per_tree_mean": round(float(np.mean(leaves)), 2),
            "objective": cfg["objective"]["name"],
            "params": {k: v for k, v in dict(XGB_PARAMS, **(params or {})).items() if k != "n_jobs"},
            "gain_importance": dict(sorted(imp.items(), key=lambda t: -t[1]))}


def _leaves(node):
    if "leaf" in node:
        yield node
    for ch in node.get("children", []):
        yield from _leaves(ch)


if __name__ == "__main__":                     # the child process started by run_isolated
    import pickle
    import sys
    name, args = pickle.loads(open(sys.argv[1], "rb").read())
    result = globals()[name](*args)
    with open(sys.argv[2], "wb") as fh:
        pickle.dump(result, fh)
