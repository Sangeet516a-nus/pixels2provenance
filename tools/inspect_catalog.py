#!/usr/bin/env python
"""Summarise the final catalogue.

    python tools/inspect_catalog.py                                   # outputs/stage8_catalogue/catalog.csv
    python tools/inspect_catalog.py path/to/catalog.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main(path="outputs/stage8_catalogue/catalog.csv"):
    cat = pd.read_csv(path)
    print(f"{path}: {len(cat):,} rows x {cat.shape[1]} columns\n")
    print("=== rows per population ===")
    print(cat.classification.value_counts().to_string(), "\n")
    print("=== rows per split (buffer = inside a collar) ===")
    print(cat.split_tag.fillna("buffer").value_counts().to_string(), "\n")
    if "cr_candidate" in cat:
        print("=== cosmic-ray checks (Stage 1c) ===")
        print(f"  cosmic-ray candidates  : {int((cat.cr_candidate.astype(str) == 'True').sum())} "
              f"(no other band covers: {int(cat.cr_candidate.isna().sum())})")
        print(f"  footprints with masked cosmic-ray pixels: {int((cat.cr_pixels_masked > 0).sum())}")
    if "classification_override" in cat:
        print("=== classification differs from the cluster name (Stage 2e) ===")
        print(cat.classification_override.fillna("(cluster name kept)").value_counts().to_string(), "\n")
    gal = cat[(cat.classification == "Galaxy") & (cat.anomaly_flag == 0)]
    pc = ["p_smooth", "p_spiral", "p_edgeon", "p_irregular", "p_ambiguous"]
    print("=== galaxies ===")
    print(f"  probabilities sum to 1 : {np.allclose(gal[pc].sum(axis=1), 1, atol=1e-4)}")
    print(f"  label heritage         : {gal.is_pseudo_label.astype(str).value_counts().to_dict()}  "
          "(False = GZH label, True = model inference only)")
    print("  predicted classes      :", gal.pred_class.value_counts().to_dict())
    pm = gal.p_merger.astype(float)
    print(f"  merger probability     : median {pm.median():.3f}, range {pm.min():.3f} - {pm.max():.3f}")
    lab = gal[gal.gz_label.notna()]
    if len(lab):
        print(f"  prediction = GZH label on {(lab.gz_label == lab.pred_class).mean():.1%} of {len(lab)} labelled "
              "galaxies (includes training rows: NOT a test score - see stage6a_test_eval/test_metrics.json)")
    print("\n=== sample rows ===")
    show = ["designation", "classification", "split_tag", "gz_label", "pred_class", "p_merger", "pred_entropy",
            "mag", "snr", "gini", "m20", "sersic_n"]
    print(gal[[c for c in show if c in gal]].head(10).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
