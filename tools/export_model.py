#!/usr/bin/env python
"""Export the trained models of a run as a MODEL BUNDLE, so fields without Galaxy Zoo labels can be classified.

    python tools/export_model.py --name goods_s_2026-10-24                  # from outputs/
    python tools/export_model.py --name goods_s --outputs runs/2026-10-25_101500/outputs

Writes models/<name>/ with the Stage 4b/4c files (the final-ensemble members: fine-tuned ResNet-18 / EfficientNet-B0
weights, physical XGBoost, fusion XGBoost; the merger XGBoost, the CNN PCA, the ensemble member list and CNN input
settings in fusion_meta.json, temperature and Platt calibration) and model_card.json (bands, training field, label
counts, dev/test scores). Use it with
supervised.mode: apply and supervised.model_bundle: models/<name> (tools/run_field.py and the app do this).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outputs", default="outputs", help="the run's outputs folder")
    ap.add_argument("--dest", default="models")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    from p2p.context import Ctx, load_json
    from p2p.modes import BUNDLE_DIRS, BUNDLE_FILES
    out = Path(a.outputs) if Path(a.outputs).is_absolute() else ROOT / a.outputs
    run_log = load_json(out / "run_log.json") if (out / "run_log.json").exists() else {}
    for k in ("4b", "4c"):
        st = run_log.get(k, {}).get("summary", {})
        if isinstance(st, dict) and st.get("mode") == "apply":
            sys.exit(f"{out} is itself an apply-mode run (stage {k} copied a bundle) - export from a TRAINING run.")
    dest = (ROOT / a.dest / a.name)
    if dest.exists():
        sys.exit(f"{dest} exists - choose another --name")
    # the files of the final-ensemble members are required only for the members the run actually used
    members = load_json(out / BUNDLE_DIRS["4b"] / "fusion_meta.json").get("ensemble_members") or ["xgb_fusion"] \
        if (out / BUNDLE_DIRS["4b"] / "fusion_meta.json").exists() else []
    member_file = {"xgb_physical": "xgb_physical.joblib", "resnet18_ft": "finetuned_resnet18.pt",
                   "efficientnet_b0_ft": "finetuned_efficientnet_b0.pt"}
    optional = {"dev_metrics.json", "xgb_merger.joblib", "cnn_pca.joblib"} | \
        {f for m, f in member_file.items() if m not in members}
    for k, d in BUNDLE_DIRS.items():
        (dest / d).mkdir(parents=True, exist_ok=True)
        for f in BUNDLE_FILES[k]:
            src = out / d / f
            if not src.exists() and f not in optional:
                shutil.rmtree(dest)
                sys.exit(f"missing {src} - run Stages 4a-4c first")
            if src.exists():
                shutil.copy2(src, dest / d / f)
    pts = load_json(out / "stage1a_acquisition" / "pointings.json")
    lab = load_json(out / "stage3a_gzh_labels" / "label_summary.json")
    train_counts = {c: int(v.get("train", 0)) for c, v in lab.get("per_split_class", {}).items()}
    # training-set feature medians, for the covariate-shift table of Stage 7c in apply mode
    ctx = Ctx(ROOT, a.config)
    ctx.out = out
    from p2p.features import supervised_splits
    tr = supervised_splits(ctx)["train"]
    med = {c: float(pd.to_numeric(tr[c], errors="coerce").median())
           for c in ("snr", "mag", "r_half", "gini", "asymmetry", "concentration") if c in tr}
    test = load_json(out / "stage6a_test_eval" / "test_metrics.json") if (out / "stage6a_test_eval" / "test_metrics.json").exists() else {}
    card = {"name": a.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trained_on": {p: {"centre": v.get("centre"), "files": {b: Path(f).name for b, f in v.get("bands", {}).items()}}
                           for p, v in pts["pointings"].items()},
            "bands": pts["bands"], "train_class_counts": train_counts, "n_train_labels": int(sum(train_counts.values())),
            "train_feature_medians": med,
            "dev_metrics": {k: v for k, v in load_json(out / "stage4b_xgboost_fusion" / "dev_metrics.json").items()
                            if k in ("accuracy", "macro_f1", "macro_f1_95ci", "balanced_acc", "nll", "n")},
            "calibration": load_json(out / "stage4c_calibration" / "temperature.json"),
            "test_metrics": {k: test.get(k) for k in ("macro_f1", "macro_f1_ci95", "balanced_acc", "ece", "n")},
            "source_outputs": out.name if not str(out).startswith(str(ROOT)) else str(out.relative_to(ROOT))}
    (dest / "model_card.json").write_text(json.dumps(card, indent=1))
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"model bundle -> {dest.relative_to(ROOT)} ({size:.0f} MB); bands {card['bands']}, "
          f"{card['n_train_labels']} training labels")


if __name__ == "__main__":
    main()
