"""XGBoost in its own process.

On macOS, PyTorch and XGBoost each load their own OpenMP runtime (libomp). Two
OpenMP runtimes in one Python process can crash it with a segmentation fault,
which no try/except can catch. So every XGBoost call - fit, predict, TreeSHAP -
runs in a short-lived child process that never imports torch. Only the trained
booster (raw JSON bytes) travels back to the main process.

This module must NOT import torch (directly or through p2p.nets / p2p.inference).
"""
from __future__ import annotations

import numpy as np

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                  eval_metric="mlogloss", n_jobs=2)


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


def fit(X, y, w, Xdev, ydev, seed):
    from xgboost import XGBClassifier
    kw = dict(XGB_PARAMS, random_state=seed)
    if Xdev is not None and len(Xdev):
        model = XGBClassifier(**kw, early_stopping_rounds=30)
        model.fit(X, y, sample_weight=w, eval_set=[(Xdev, ydev)], verbose=False)
    else:
        model = XGBClassifier(**kw).fit(X, y, sample_weight=w)
    best = getattr(model, "best_iteration", None)
    return bytes(model.get_booster().save_raw("json")), (None if best is None else int(best))


def _booster(raw):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(bytearray(raw))
    return b, xgb


def predict(raw, best, X):
    b, xgb = _booster(raw)
    rng = (0, best + 1) if best is not None else (0, 0)
    p = np.asarray(b.predict(xgb.DMatrix(np.asarray(X, np.float32)), iteration_range=rng))
    return np.column_stack([1 - p, p]) if p.ndim == 1 else p          # binary -> two columns


def contribs(raw, best, X):
    """Exact TreeSHAP values, shape (n, classes, features) without the bias column."""
    b, xgb = _booster(raw)
    rng = (0, best + 1) if best is not None else (0, 0)
    c = np.asarray(b.predict(xgb.DMatrix(np.asarray(X, np.float32)), pred_contribs=True, iteration_range=rng))
    if c.ndim == 2:                                                   # binary: (n, features + 1)
        c = c[:, None, :]
    return c[:, :, :-1]


if __name__ == "__main__":                     # the child process started by run_isolated
    import pickle
    import sys
    name, args = pickle.loads(open(sys.argv[1], "rb").read())
    result = globals()[name](*args)
    with open(sys.argv[2], "wb") as fh:
        pickle.dump(result, fh)
