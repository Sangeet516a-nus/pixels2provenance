"""Compare ensemble member weights on the saved out-of-fold predictions of Stage 4b. No network is trained and neither
the test nor the external galaxies are read.

    python tools/reweight_ensemble.py                                   # equal vs 0.25 / 0.25 / 0.5, nothing written
    python tools/reweight_ensemble.py --weights resnet18_ft=0.2 efficientnet_b0_ft=0.3 xgb_physical=0.5
    python tools/reweight_ensemble.py --apply                           # writes them into the Stage 4b outputs

Printed: Macro-F1 and log-loss of both, the paired bootstrap difference, per-class F1, and weights fitted by log-loss in
nested cross-validation (each fold's weights come from the other folds): if those stay close to the chosen weights
and score about the same, the choice is stable. After --apply set the same weights in
supervised.ensemble_member_weights (a later Stage 4b reads them from there) and run  python run_pipeline.py --from 4c
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p2p.context import Ctx, load_json, save_json  # noqa: E402
from p2p.inference import combine_members  # noqa: E402
from p2p.nets import MORPH_CLASSES, nll  # noqa: E402

DEFAULT = {"resnet18_ft": 0.25, "efficientnet_b0_ft": 0.25, "xgb_physical": 0.5}


def macro_f1(y, P):
    return f1_score(y, P.argmax(1), average="macro")


def nested_weights(y, parts, fold, step=0.05):
    """Weights fitted by minimum log-loss on the other folds, applied to each fold in turn."""
    names = list(parts)
    n = int(round(1 / step))
    grid = [dict(zip(names, np.array(w) * step)) for w in itertools.product(range(n + 1), repeat=len(names)) if sum(w) == n]
    out, chosen = np.zeros_like(parts[names[0]]), []
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        best = min(grid, key=lambda w: nll(y[tr], combine_members({m: parts[m][tr] for m in names}, w)))
        chosen.append({m: round(float(v), 2) for m, v in best.items()})
        out[va] = combine_members({m: parts[m][va] for m in names}, best)
    return out, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", help="member=weight ..., default 0.25 / 0.25 / 0.5")
    ap.add_argument("--apply", action="store_true", help="write the weights into oof_fusion.npz and fusion_meta.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    ctx = Ctx(ROOT, args.config)
    meta_path = ctx.require("4b", "fusion_meta.json")
    meta = load_json(meta_path)
    members = list(meta["ensemble_members"])
    weights = {k: float(v) for k, v in (w.split("=") for w in args.weights)} if args.weights else DEFAULT
    if set(weights) != set(members):
        sys.exit(f"--weights must name exactly the ensemble members {members}")
    mo = np.load(ctx.require("4b", "oof_members.npz"))
    d = np.load(ctx.require("4b", "oof_fusion.npz"))
    y, fold = d["y"].astype(int), d["fold"]
    parts = {m: mo[m] for m in members}
    eq, wt = combine_members(parts), combine_members(parts, weights)

    rng = np.random.default_rng(ctx.seed)
    diff = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        diff.append(macro_f1(y[i], wt[i]) - macro_f1(y[i], eq[i]))
    print(f"out-of-fold, n = {len(y)}, members {members}")
    print(f"  equal weights    Macro-F1 {macro_f1(y, eq):.4f}  log-loss {nll(y, eq):.4f}")
    print(f"  {weights}\n                   Macro-F1 {macro_f1(y, wt):.4f}  log-loss {nll(y, wt):.4f}")
    print(f"  difference       {macro_f1(y, wt) - macro_f1(y, eq):+.4f} (paired 95 % CI "
          f"{np.percentile(diff, 2.5):+.4f} to {np.percentile(diff, 97.5):+.4f})")
    f_eq = f1_score(y, eq.argmax(1), average=None, labels=range(len(MORPH_CLASSES)))
    f_wt = f1_score(y, wt.argmax(1), average=None, labels=range(len(MORPH_CLASSES)))
    print("  per class F1     " + ", ".join(f"{c} {a:.3f} -> {b:.3f}" for c, a, b in zip(MORPH_CLASSES, f_eq, f_wt)))
    nest, chosen = nested_weights(y, parts, fold)
    print(f"  fitted in nested CV: Macro-F1 {macro_f1(y, nest):.4f}, log-loss {nll(y, nest):.4f}; per fold {chosen}")
    if args.apply:
        np.savez(ctx.path("4b", "oof_fusion.npz"), **{k: (wt if k == "morph" else d[k]) for k in d.files})
        meta["ensemble_member_weights"] = weights
        save_json(meta, meta_path)
        print("Written. Set the same weights in supervised.ensemble_member_weights, then: "
              "python run_pipeline.py --from 4c")


if __name__ == "__main__":
    main()
