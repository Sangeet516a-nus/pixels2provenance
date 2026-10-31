"""Train vs apply mode of the supervised part (Stages 3a-7c).

train (default): the field has Galaxy Zoo: Hubble labels; Stages 4a-4d train the models on its pooled train + dev
                 galaxies (5-fold cross-validation), calibrate on the out-of-fold predictions and test once on
                 test (the original design).
apply          : the field has no (or too few) GZH labels - most HST fields outside the GZH surveys.
                 Stage 4a extracts the frozen CNN features as usual; Stages 4b-4c copy a MODEL BUNDLE (the XGBoost
                 models, ensemble weights and calibration) exported
                 from a training run (tools/export_model.py), Stage 4d is skipped, and every galaxy is
                 classified by that frozen model in Stage 8. If the field does have some GZH or CANDELS labels,
                 Stages 6a and 7a-7c use ALL of them as an out-of-field test (the model never saw this field).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# Stage 4a needs no bundle: its CNNs are the frozen ImageNet networks (only the band set must match, check_bands).
BUNDLE_FILES = {"4b": ["xgb_morphology.joblib", "xgb_merger.joblib", "cnn_pca.joblib", "fusion_meta.json", "dev_metrics.json",
                       "cv_metrics.json", "xgb_physical.joblib", "finetuned_resnet18.pt", "finetuned_efficientnet_b0.pt"],
                "4c": ["temperature.json"]}
BUNDLE_DIRS = {"4b": "stage4b_xgboost_fusion", "4c": "stage4c_calibration"}
# predictions of the TRAINING field, keyed by object_id: in apply mode they would be matched to the new field's
# galaxies by id, so they are removed (the fine-tuned networks then predict the new galaxies themselves)
TRAINING_ONLY_FILES = {"4b": ["finetuned_probs.npz", "oof_members.npz", "oof_fusion.npz"]}


def is_apply(ctx) -> bool:
    return str(ctx.cfg.get("supervised", {}).get("mode", "train")).lower() == "apply"


def bundle_dir(ctx) -> Path:
    b = ctx.cfg.get("supervised", {}).get("model_bundle")
    if not b:
        raise RuntimeError("supervised.mode is 'apply' but supervised.model_bundle is empty in config.yaml. "
                           "Export one from a training run: python tools/export_model.py --name <name>")
    p = Path(b)
    p = p if p.is_absolute() else ctx.root / p
    if not (p / "model_card.json").exists():
        raise RuntimeError(f"model bundle {p} not found (no model_card.json). "
                           "Export one from a training run: python tools/export_model.py --name <name>")
    return p


def card(ctx) -> dict:
    return json.loads((bundle_dir(ctx) / "model_card.json").read_text())


def install_bundle(ctx, key: str) -> dict:
    """Copy the bundle's files for stage `key` (4b or 4c) into this run's stage folder."""
    src = bundle_dir(ctx) / BUNDLE_DIRS[key]
    out = ctx.stage_dir(key)
    copied = []
    for f in BUNDLE_FILES[key]:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
            copied.append(f)
        elif (out / f).exists():
            (out / f).unlink()             # a leftover of an earlier training run must not mix with the bundle
    for f in TRAINING_ONLY_FILES.get(key, []):
        if (out / f).exists():
            (out / f).unlink()
    c = card(ctx)
    where = "; ".join(f"{p} at RA {v['centre'][0]:.3f} Dec {v['centre'][1]:+.3f}" for p, v in c.get("trained_on", {}).items()
                      if v.get("centre"))
    ctx.log.info(f"[Stage {key}] APPLY mode: frozen model bundle '{c.get('name')}' (trained on {where or '?'}, "
                 f"{c.get('n_train_labels', '?')} labels, bands {c.get('bands')}) -> {', '.join(copied)}")
    return {"mode": "apply", "bundle": c.get("name"), "files": copied}


def cnn_input_settings(ctx) -> dict:
    """The supervised settings that shape the CNN input (crop, neighbour mask, cosmic-ray cleaning, rotations,
    input size). In apply mode they come from the bundle's fusion_meta.json: the XGBoost models (and the CNN PCA)
    were fitted on features made that way, so features made any other way would not match them."""
    cfg = dict(ctx.cfg.get("supervised", {}))
    if is_apply(ctx):
        p = bundle_dir(ctx) / BUNDLE_DIRS["4b"] / "fusion_meta.json"
        if p.exists():
            meta = json.loads(p.read_text())
            cfg.update({k: v for k, v in (meta.get("cnn_input") or {}).items() if v is not None})
            if meta.get("input_size"):
                cfg["input_size"] = int(meta["input_size"])
    return cfg


def check_bands(ctx):
    """The CNN input (R, G, B) and the colours expect the bundle's band set, in the same order."""
    from .context import load_json
    bands = load_json(ctx.require("1a", "pointings.json"))["bands"]
    want = card(ctx).get("bands", [])
    if len(bands) != len(want):
        raise RuntimeError(f"apply mode: this field has {len(bands)} bands {bands}, the model was trained on "
                           f"{len(want)} bands {want}. Download the same number of filters (see tools/run_field.py).")
    if bands != want:
        ctx.log.info(f"[apply] NOTE: bands {bands} stand in for the model's {want} (channel by channel, blue to red). "
                     "Colours and the image branch see slightly different passbands: treat the result as approximate.")
