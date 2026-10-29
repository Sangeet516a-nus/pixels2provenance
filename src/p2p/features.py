"""Tabular pre-processing, the ImageNet input of the frozen CNNs, and the galaxy / label tables."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .utils import signed_log1p


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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def imagenet_input(cubes, size=224, q_lo=1.0, q_hi=99.5, soft=0.1):
    """PSF-matched cutouts (N, 3, 128, 128), bands blue -> red, in e-/s -> ImageNet-ready tensor (N, 3, size, size).
      1. one asinh stretch per cutout, the SAME for all three bands (percentiles of all its pixels), so the colour
         of the galaxy survives (a separate stretch per band would make every galaxy grey);
      2. channels R, G, B = reddest, middle, bluest band (F775W, F606W, F435W); a band without data stays 0;
      3. ImageNet mean / std per channel (the statistics the frozen BatchNorm layers expect);
      4. bilinear resize to size x size (224 = the image size the networks were trained on)."""
    import torch
    import torch.nn.functional as F
    x = np.nan_to_num(np.asarray(cubes, np.float32))[:, ::-1]                 # blue->red  ->  R, G, B
    out = np.empty_like(x)
    for n in range(len(x)):
        lo, hi = np.percentile(x[n], [q_lo, q_hi])
        v = np.clip((x[n] - lo) / max(hi - lo, 1e-8), 0, 1)
        out[n] = np.arcsinh(v / soft) / np.arcsinh(1.0 / soft)
    t = torch.from_numpy((out - IMAGENET_MEAN) / IMAGENET_STD)
    if size != t.shape[-1]:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def physical_matrix(df, columns):
    """The 14 physical (Pass-2) features as they are: trees need no scaling, and NaN (not measurable) is
    handled natively by XGBoost."""
    return df.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


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
    from .modes import is_apply
    if is_apply(ctx):
        # the frozen model never saw this field: EVERY labelled galaxy (any split, buffer too) is test data
        from .context import StageSkipped
        allg = df[~df.is_pseudo_label & ~df.edge_flag]
        if ctx.cfg["frame_qc"].get("exclude_flagged"):
            allg = allg[allg.frame_quality_flag == 1]
        if allg.empty:
            raise StageSkipped("apply mode: no GZH-labelled galaxies in this field, so no in-field evaluation "
                               "(every galaxy is still classified in Stage 8)")
        allg = allg.assign(gz_label_id=allg.gz_label_id.astype(int)).reset_index(drop=True)
        return {"train": allg.iloc[0:0], "dev": allg.iloc[0:0], "test": allg}
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
