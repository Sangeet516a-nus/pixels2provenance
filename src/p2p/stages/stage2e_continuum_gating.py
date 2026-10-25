"""Stage 2e - Continuum handling and outlier gating.

Purpose : do not throw away real galaxies that sit between populations, and do
          not silently drop true anomalies either.
Method  : GMM components are summed into Star / Galaxy / Noise posteriors using
          the Stage-2d names. An object that HDBSCAN calls noise (-1) but whose
          galaxy posterior is >= 0.80 is KEPT as a galaxy and tagged
          is_transitional. An HDBSCAN-noise object with no posterior above 0.50
          AND a CAE reconstruction error above the 99th percentile of the train
          split is an extreme outlier -> anomaly_log.csv (anomaly_flag = 1).
          Physical consistency gate (gating.physical_gate): a GMM cluster is
          named as a whole, but one cluster can mix stars with compact galaxies.
          Each object is then checked against evidence the cluster name must
          not contradict:
            cosmic-ray candidate (no second band, Stage 1c)  -> Noise
            Gaia star                                         -> Star
            in a Star cluster without star evidence (Gaia, or the bright stellar
              locus of Stage 2d): sharper than the PSF -> Noise, otherwise -> Galaxy
              (faint point-like sources in a deep field are mostly compact galaxies)
            in a Galaxy cluster but on the stellar locus (Stage 2d)                    -> Star
            in a Noise cluster but confirmed in two bands, SNR >= 5, not sharper than
              the PSF (a real source): on the stellar locus -> Star, otherwise -> Galaxy
          The reason is kept in classification_override (empty = cluster name
          kept); cluster_semantic_2d keeps the pure cluster name, which is what
          Stage 2d scores for Objective 2.
          If HDBSCAN failed (Stage 2b), the transitional rule cannot be applied:
          is_transitional is left EMPTY (unknown, not False), and the outlier
          rule runs without the HDBSCAN condition.
Output  : segregation.csv (final cluster_semantic for every object), anomaly_log.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_json, load_objects
from ..utils import as_bool

NAMES = ("Star", "Galaxy", "Noise")


def physical_gate(df, sem, override, cfg_b):
    """Object-level checks a cluster name must not contradict (see the module docstring)."""
    tol = cfg_b["point_source_tolerance"]
    ratio = df.fwhm_over_psf.to_numpy(float)
    cr = as_bool(df.cr_candidate).to_numpy() if "cr_candidate" in df else np.zeros(len(df), bool)
    gaia = as_bool(df.gaia_hit).to_numpy() if "gaia_hit" in df else np.zeros(len(df), bool)
    locus = as_bool(df.stellar_locus).to_numpy() if "stellar_locus" in df else np.zeros(len(df), bool)
    two_band = df.cr_candidate.map(lambda v: str(v).strip().lower() in ("false", "0", "0.0")).to_numpy() \
        if "cr_candidate" in df else np.ones(len(df), bool)
    resolved = np.nan_to_num(ratio, nan=0) > 1 + 2 * tol
    real_noise_cluster = two_band & (df.snr.to_numpy(float) >= 5) & (np.nan_to_num(ratio, nan=0) >= cfg_b["sub_psf_factor"])
    sem, override = sem.copy(), override.copy()

    def move(mask, to, why):
        mask = mask & (sem != to)
        sem[mask] = to
        override[mask] = why

    move((sem == "Noise") & real_noise_cluster & locus, "Star", "Noise cluster, but confirmed in 2 bands on the stellar locus")
    move((sem == "Noise") & real_noise_cluster & ~locus, "Galaxy",
         "Noise cluster, but a real source (confirmed in 2 bands, SNR >= 5, not sharper than the PSF)")
    move((sem == "Galaxy") & locus & ~resolved, "Star", "Galaxy cluster, but on the stellar locus")
    broader = np.nan_to_num(ratio, nan=0) > 1 + tol
    sharper = np.nan_to_num(ratio, nan=1) < cfg_b["sub_psf_factor"]
    no_star_evidence = (sem == "Star") & ~locus & ~gaia
    move(no_star_evidence & broader, "Galaxy", "Star cluster, but broader than a point source")
    move(no_star_evidence & sharper, "Noise", "Star cluster, but sharper than the PSF")
    move(no_star_evidence & ~broader & ~sharper, "Galaxy",
         "Star cluster, but too faint for size to separate a star from a compact galaxy")
    move(gaia, "Star", "Gaia star")
    move(cr & ~gaia, "Noise", "cosmic-ray candidate (no second band confirms it)")
    return sem, override


def run(ctx: Ctx):
    cfg = ctx.cfg["gating"]
    out = ctx.stage_dir("2e")
    df = load_objects(ctx, ["sources", "pass1", "clusters", "gaia", "evidence"])
    names = {int(k): v for k, v in load_json(ctx.require("2d", "cluster_names.json")).items()}
    err = pd.Series(np.load(ctx.require("2a", "recon_error.npy")),
                    index=np.load(ctx.require("2a", "object_id.npy"))).reindex(df.object_id).to_numpy()

    post = {n: np.zeros(len(df)) for n in NAMES}
    for k, n in names.items():
        col = f"gmm_p{k}"
        if col in df and n in post:
            post[n] += df[col].to_numpy(float)
    hdb = df.cluster_id_hdbscan.to_numpy(float)
    hdb_ok = np.isfinite(hdb).any() and (hdb >= 0).any()
    if not hdb_ok:
        ctx.log.info("[Stage 2e] WARNING: HDBSCAN failed (Stage 2b) - is_transitional left empty; the outlier "
                     "rule uses GMM posterior + reconstruction error only.")
    hnoise = (hdb == -1) if hdb_ok else np.ones(len(df), bool)
    P = np.column_stack([post[n] for n in NAMES])
    thr = float(np.nanpercentile(err[(df.split_tag == "train").to_numpy()], cfg["recon_percentile"]))
    transitional = hnoise & (post["Galaxy"] >= cfg["transitional_p"]) if hdb_ok else np.zeros(len(df), bool)
    anomaly = hnoise & (P.max(1) < cfg["outlier_max_p"]) & (np.nan_to_num(err, nan=-1) > thr)

    sem = df.cluster_semantic_2d.to_numpy(object).copy()
    sem[transitional] = "Galaxy"
    override = np.full(len(df), None, dtype=object)
    override[transitional] = "transitional (HDBSCAN noise, GMM galaxy posterior >= transitional_p)"
    if cfg.get("physical_gate", True):
        sem, override = physical_gate(df, sem, override, ctx.cfg["baseline"])
        ctx.log.info(f"[Stage 2e] physical gate: {pd.Series(override).value_counts().to_dict()}")
    res = pd.DataFrame({"object_id": df.object_id, "cluster_semantic": sem,
                        "gmm_p_star": post["Star"].round(4), "gmm_p_galaxy": post["Galaxy"].round(4),
                        "gmm_p_noise": post["Noise"].round(4), "reconstruction_error": err,
                        "is_transitional": transitional if hdb_ok else None, "anomaly_flag": anomaly.astype(int),
                        "classification_override": override})
    res.to_csv(out / "segregation.csv", index=False)
    log = res[res.anomaly_flag == 1].merge(df[["object_id", "ra", "dec", "pointing", "cluster_id_gmm",
                                               "cluster_id_hdbscan"]], on="object_id")
    log["reason"] = "HDBSCAN noise + no GMM population > 0.5 + reconstruction error above train 99th pct"
    log.to_csv(out / "anomaly_log.csv", index=False)
    ctx.log.info(f"[Stage 2e] populations {res.cluster_semantic.value_counts().to_dict()} | transitional kept "
                 f"{int(transitional.sum())} | anomalies logged {int(anomaly.sum())} (recon threshold {thr:.5f})")
    return {"populations": res.cluster_semantic.value_counts().to_dict(), "anomalies": int(anomaly.sum())}
