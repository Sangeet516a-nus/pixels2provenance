"""Shared run context: configuration, folders, logging and the stage registry.

Every stage module exposes ``run(ctx)``. A stage reads what earlier stages
wrote (through ``ctx.require`` / ``load_objects``) and writes its own files
into ``outputs/<stage folder>/`` so each step can be inspected on its own.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# (key, module name, output folder, title) in EXECUTION order.
# Phase 5 runs before Phase 4 so the supervised models train on the
# PSF-refined Pass-2 features they are deployed on.
STAGES = [
    ("0",  "stage0_partition",        "stage0_partition",        "Sky-region partitioning"),
    ("1a", "stage1a_acquisition",     "stage1a_acquisition",     "Archive acquisition / local scan"),
    ("1b", "stage1b_frame_qc",        "stage1b_frame_qc",        "Frame quality control (CAE)"),
    ("1c", "stage1c_detection",       "stage1c_detection",       "Detection, deblending, aligned cutouts"),
    ("1d", "stage1d_psf_match",       "stage1d_psf_match",       "Cross-filter PSF homogenisation"),
    ("1e", "stage1e_pass1_morphology", "stage1e_pass1_morphology", "Pass-1 photometry and morphology"),
    ("2a", "stage2a_representation",  "stage2a_representation",  "Representation learning + fusion"),
    ("2b", "stage2b_clustering",      "stage2b_clustering",      "Unsupervised clustering"),
    ("2c", "stage2c_baseline",        "stage2c_baseline",        "Baseline segregation + Gaia match"),
    ("2d", "stage2d_semantic_audit",  "stage2d_semantic_audit",  "Semantic audit"),
    ("2e", "stage2e_continuum_gating", "stage2e_continuum_gating", "Continuum handling + outlier gating"),
    ("3a", "stage3a_gzh_labels",      "stage3a_gzh_labels",      "GZH label anchoring"),
    ("3b", "stage3b_star_routing",    "stage3b_star_routing",    "Star calibration routing"),
    ("3c", "stage3c_noise_model",     "stage3c_noise_model",     "Noise false-positive model"),
    ("5a", "stage5a_empirical_psf",   "stage5a_empirical_psf",   "Empirical PSF construction"),
    ("5b", "stage5b_pass2_refinement", "stage5b_pass2_refinement", "Pass-2 morphology refinement"),
    ("4a", "stage4a_dual_branch",     "stage4a_dual_branch",     "Dual-branch multi-task training"),
    ("4b", "stage4b_ensemble",        "stage4b_ensemble",        "Ensemble consensus"),
    ("4c", "stage4c_calibration",     "stage4c_calibration",     "Probability calibration"),
    ("4d", "stage4d_ablation",        "stage4d_ablation",        "Controlled ablation"),
    ("6a", "stage6a_test_eval",       "stage6a_test_eval",       "In-domain test (run once)"),
    ("6b", "stage6b_uncertainty",     "stage6b_uncertainty",     "Uncertainty audit"),
    ("6c", "stage6c_xai",             "stage6c_xai",             "Explainable AI (Grad-CAM, SHAP)"),
    ("7a", "stage7a_taxonomy",        "stage7a_taxonomy",        "Taxonomy alignment + ceiling"),
    ("7b", "stage7b_crosscat",        "stage7b_crosscat",        "Cross-catalogue evaluation"),
    ("7c", "stage7c_generalisation",  "stage7c_generalisation",  "Generalisation disentanglement"),
    ("8",  "stage8_catalogue",        "stage8_catalogue",        "Unified catalogue (catalog.csv)"),
]
STAGE_DIR = {k: d for k, _, d, _ in STAGES}


class MissingInput(RuntimeError):
    """Raised when a stage needs a file that an earlier stage has not written yet."""


class StageSkipped(RuntimeError):
    """Raised by a stage that cannot run with the data provided (e.g. no CANDELS table)."""


class Ctx:
    def __init__(self, root: str | Path, config_file: str = "config.yaml"):
        self.root = Path(root).resolve()
        with open(self.root / config_file) as f:
            self.cfg = yaml.safe_load(f)
        p = self.cfg["paths"]
        self.raw = self.root / p["raw"]
        self.catalogs = self.root / p["catalogs"]
        self.out = self.root / p["outputs"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = _make_logger(self.out / "pipeline.log")
        self.seed = int(self.cfg["project"].get("seed", 42))
        self.set_seed()

    # ---------------- folders and files ----------------
    def stage_dir(self, key: str) -> Path:
        d = self.out / STAGE_DIR[key]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, key: str, name: str) -> Path:
        return self.out / STAGE_DIR[key] / name

    def require(self, key: str, name: str) -> Path:
        p = self.path(key, name)
        if not p.exists():
            raise MissingInput(f"{self.rel(p)} not found - run stage {key} first "
                               f"(python run_pipeline.py --only {key}).")
        return p

    # ---------------- small helpers ----------------
    def rel(self, p) -> str:
        """Path relative to the project folder when possible (outputs may live elsewhere)."""
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except Exception:
            pass

    @property
    def n_jobs(self) -> int:
        n = int(self.cfg["project"].get("n_jobs", -1))
        return (os.cpu_count() or 1) if n < 0 else max(1, n)

    def device(self):
        import torch
        want = str(self.cfg["project"].get("device", "auto")).lower()
        if want == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(want)


# ---------------- JSON / table helpers ----------------
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# Which file holds which per-object table. load_objects() merges them on object_id.
OBJECT_TABLES = {
    "sources":     ("1c", "sources.csv"),
    "pass1":       ("1e", "features_pass1.csv"),
    "clusters":    ("2b", "clusters.csv"),
    "baseline":    ("2c", "baseline.csv"),
    "gaia":        ("2c", "gaia_match.csv"),
    "evidence":    ("2d", "evidence.csv"),
    "segregation": ("2e", "segregation.csv"),
    "labels":      ("3a", "labels.csv"),
    "routing":     ("3b", "star_routing.csv"),
    "noise":       ("3c", "noise_scores.csv"),
    "pass2":       ("5b", "features_pass2.csv"),
}


def load_objects(ctx: Ctx, parts) -> pd.DataFrame:
    """Merge per-object tables on object_id (left join on the source list).
    A later part OVERWRITES columns of the same name for the rows it holds -
    this is how Pass-2 features replace Pass-1 features for the galaxy track."""
    df = None
    for part in parts:
        key, name = OBJECT_TABLES[part]
        t = pd.read_csv(ctx.require(key, name))
        if df is None:
            df = t
            continue
        overlap = [c for c in t.columns if c in df.columns and c != "object_id"]
        df = df.merge(t, on="object_id", how="left", suffixes=("", "__new"), indicator="_has")
        rows = df["_has"] == "both"
        for c in overlap:
            df.loc[rows, c] = df.loc[rows, c + "__new"]
            df = df.drop(columns=c + "__new")
        df = df.drop(columns="_has")
    return df


def _make_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("p2p")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
