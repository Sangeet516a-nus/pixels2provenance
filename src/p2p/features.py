"""Tabular pre-processing (fit on TRAIN only) and model-ready tensors."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .utils import asinh_stack, signed_log1p


class TabularPrep:
    """Median imputation -> signed log -> RobustScaler. Columns that are entirely
    NaN in the training data (e.g. colours when only one band exists) are dropped."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.kept = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()

    def fit(self, df: pd.DataFrame):
        cols = [c for c in self.columns if c in df.columns]
        X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        keep = ~np.all(np.isnan(X), axis=0)
        self.kept = [c for c, k in zip(cols, keep) if k]
        X = signed_log1p(self.imputer.fit_transform(df[self.kept].apply(pd.to_numeric, errors="coerce")))
        self.scaler.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.kept).apply(pd.to_numeric, errors="coerce")
        return self.scaler.transform(signed_log1p(self.imputer.transform(X))).astype(np.float32)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TabularPrep":
        return joblib.load(path)


def load_cutouts(ctx):
    """PSF-matched cutouts written by Stage 1d (memory-mapped, row = cutout_index)."""
    return np.load(ctx.require("1d", "cutouts.npy"), mmap_mode="r")


def image_tensor(cutouts, index):
    import torch
    batch = np.asarray(cutouts[np.asarray(index, int)], dtype=np.float32)
    return torch.tensor(asinh_stack(batch))


def make_tensors(df: pd.DataFrame, cutouts, prep: TabularPrep, with_labels=True):
    import torch
    d = {"img": image_tensor(cutouts, df["cutout_index"].to_numpy()),
         "tab": torch.tensor(prep.transform(df)),
         "object_id": df["object_id"].to_numpy()}
    if with_labels and "gz_label_id" in df and df["gz_label_id"].notna().all():
        d["y"] = torch.tensor(df["gz_label_id"].astype(int).to_numpy())
        d["ym"] = torch.tensor(df["gz_merger_flag"].fillna(0).astype(float).to_numpy(), dtype=torch.float32)
    return d


def galaxy_table(ctx):
    """Every galaxy-track object with Pass-2 features (overwriting Pass 1) and labels."""
    from .context import load_objects
    from .utils import as_bool
    df = load_objects(ctx, ["sources", "pass1", "segregation", "labels", "pass2"])
    df = df[(df.cluster_semantic == "Galaxy") & (df.anomaly_flag == 0)].copy()
    df["is_pseudo_label"] = as_bool(df.is_pseudo_label.fillna(True))
    df["edge_flag"] = as_bool(df.edge_flag)
    return df


def supervised_splits(ctx):
    """GZH-labelled galaxies per split. Truncated (edge_flag) sources are excluded,
    and QC-flagged ones too when frame_qc.exclude_flagged is true."""
    df = galaxy_table(ctx)
    lab = df[~df.is_pseudo_label & df.split_tag.isin(["train", "dev", "test"]) & ~df.edge_flag]
    if ctx.cfg["frame_qc"].get("exclude_flagged"):
        lab = lab[lab.frame_quality_flag == 1]
    lab = lab.assign(gz_label_id=lab.gz_label_id.astype(int))
    splits = {s: lab[lab.split_tag == s].reset_index(drop=True) for s in ("train", "dev", "test")}
    empty = [s for s, v in splits.items() if len(v) == 0]
    if empty:
        m = df[~df.is_pseudo_label]
        per = m.split_tag.fillna("collar").value_counts().to_dict()
        per_ok = m[~m.edge_flag].split_tag.fillna("collar").value_counts().to_dict()
        raise RuntimeError(
            f"No labelled galaxies in split(s) {empty}.\n"
            f"  GZH-matched galaxies per split: {per}\n"
            f"  ... after removing edge-truncated ones: {per_ok}\n"
            "  With one pointing, Stage 0 cuts it into stripes: look at outputs/stage0_partition/split_regions.png, "
            "lower partition.collar_arcmin, or add a second pointing.")
    return splits
