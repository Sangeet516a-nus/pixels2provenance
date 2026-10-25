"""Stage 2b - Unsupervised clustering (GMM + HDBSCAN).

Purpose : group sources into populations without any label.
Method  : a Gaussian Mixture Model (diagonal covariance by default: far fewer
          parameters, so BIC can afford more, purer components) with a BIC sweep over k,
          and HDBSCAN (density-based, marks low-density points as noise -1).
          HDBSCAN is tried with a small grid of min_cluster_size values and the
          fit with the best density-based validity (DBCV) among those with >= 2
          clusters is kept; if none finds 2 clusters HDBSCAN is reported as
          FAILED and cluster_id_hdbscan is left empty (not -1, which would claim
          every object is density noise).
          Both are fitted on the TRAIN split; every other object is assigned
          with GMM.predict / hdbscan.approximate_predict - never by refitting.
Output  : clusters.csv (gmm label + soft posteriors, hdbscan label + strength),
          gmm.joblib, hdbscan.joblib, bic.png
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..context import Ctx, save_json

try:
    import hdbscan as _hdb
    _HDB = "contrib"
except Exception:
    _HDB = None


def run(ctx: Ctx):
    cfg = ctx.cfg["clustering"]
    out = ctx.stage_dir("2b")
    X = np.load(ctx.require("2a", "hybrid_features.npy"))
    oid = np.load(ctx.require("2a", "object_id.npy"))
    src = pd.read_csv(ctx.require("1c", "sources.csv")).set_index("object_id").loc[oid]
    tr = (src.split_tag == "train").to_numpy()
    Xtr = X[tr]

    # float64: in float32 the variance of a tight component can round to <= 0 ("ill-defined covariance")
    X, Xtr = X.astype(np.float64), Xtr.astype(np.float64)
    p1 = ctx.path("1e", "features_pass1.csv")
    if p1.exists():
        f = pd.read_csv(p1, usecols=["object_id", "flag"]).set_index("object_id").reindex(oid)["flag"].to_numpy()
        n_faint = int((f[tr] == 4).sum())
        ctx.log.info(f"[Stage 2b] {n_faint} of {int(tr.sum())} train sources are too faint for morphology "
                     "(their shape features are median-filled, so they sit on top of each other)")
    reg0 = float(cfg.get("reg_covar", 1e-4))
    bics, models = {}, {}
    for k in range(cfg["k_min"], cfg["k_max"] + 1):
        g, reg = None, reg0
        while g is None and reg <= 1e-2:
            try:
                g = GaussianMixture(n_components=k, covariance_type=cfg.get("covariance_type", "diag"),
                                    random_state=ctx.seed, n_init=3, reg_covar=reg).fit(Xtr)
            except ValueError:                     # a component collapsed onto (near-)identical points
                reg *= 10
        if g is None:
            ctx.log.info(f"  [GMM] k={k}  failed (collapsed component even with reg_covar 1e-2) - skipped")
            continue
        bics[k], models[k] = g.bic(Xtr), g
        ctx.log.info(f"  [GMM] k={k}  BIC={bics[k]:.1f}" + (f"  (reg_covar raised to {reg:g})" if reg > reg0 else ""))
    if not bics:
        raise RuntimeError("Every GMM fit collapsed. Lower clustering.k_max or raise clustering.reg_covar.")
    best_k = min(bics, key=bics.get)
    if best_k == cfg["k_max"]:
        ctx.log.info(f"[Stage 2b] note: BIC is lowest at the largest k tried ({best_k}); raising clustering.k_max "
                     "may split the populations further (cluster naming is many-to-one, so this is safe).")
    gmm = models[best_k]
    probs = gmm.predict_proba(X)
    res = pd.DataFrame({"object_id": oid, "cluster_id_gmm": probs.argmax(1)})
    for j in range(best_k):
        res[f"gmm_p{j}"] = probs[:, j].round(5)
    joblib.dump(gmm, out / "gmm.joblib")

    mcs = cfg["hdbscan_min_cluster_size"]
    mcs = max(10, len(Xtr) // 50) if mcs == "auto" else int(mcs)
    hdb_status, hdb_trials = "not run", []
    if _HDB:
        grid = sorted({mcs, max(5, mcs // 2), max(5, mcs // 4), 10}, reverse=True)
        best = None
        for m in grid:
            h = _hdb.HDBSCAN(min_cluster_size=m, min_samples=cfg.get("hdbscan_min_samples", 5),
                             prediction_data=True, gen_min_span_tree=True).fit(Xtr)
            n_c = len(set(h.labels_) - {-1})
            try:
                dbcv = float(h.relative_validity_) if n_c >= 2 else np.nan
            except Exception:
                dbcv = np.nan
            hdb_trials.append({"min_cluster_size": m, "clusters": n_c, "noise_fraction": round(float((h.labels_ == -1).mean()), 4),
                               "dbcv": None if not np.isfinite(dbcv) else round(dbcv, 4)})
            ctx.log.info(f"  [HDBSCAN] min_cluster_size={m}: {n_c} clusters, noise {hdb_trials[-1]['noise_fraction']:.1%}, "
                         f"DBCV {hdb_trials[-1]['dbcv']}")
            if n_c >= 2 and (best is None or np.nan_to_num(dbcv, nan=-1) > best[2]):
                best = (h, m, np.nan_to_num(dbcv, nan=-1))
        if best is not None:
            h, mcs = best[0], best[1]
            lab, strength = _hdb.approximate_predict(h, X)
            lab[tr] = h.labels_                       # training points keep their fitted labels
            res["cluster_id_hdbscan"] = lab
            res["hdbscan_strength"] = np.round(strength, 4)
            joblib.dump(h, out / "hdbscan.joblib")
            n_h = len(set(h.labels_) - {-1})
            hdb_status = "ok"
        else:
            res["cluster_id_hdbscan"] = np.nan
            n_h = 0
            hdb_status = "failed"
            ctx.log.info(f"[Stage 2b] HDBSCAN FAILED: no min_cluster_size in {grid} gives 2 clusters on this feature "
                         "space. cluster_id_hdbscan is left empty and Stage 2e gating runs without it - report "
                         "HDBSCAN as failed.")
    else:
        res["cluster_id_hdbscan"] = np.nan
        n_h = 0
        ctx.log.info("[Stage 2b] hdbscan package missing (pip install hdbscan) - HDBSCAN skipped")
    res.to_csv(out / "clusters.csv", index=False)
    save_json({"bic": bics, "best_k": best_k, "hdbscan_status": hdb_status, "hdbscan_trials": hdb_trials,
               "hdbscan_min_cluster_size": mcs, "hdbscan_clusters": n_h,
               "gmm_sizes": res.cluster_id_gmm.value_counts().to_dict()}, out / "summary.json")
    _plot(bics, best_k, out)
    ctx.log.info(f"[Stage 2b] BIC-selected k = {best_k}; GMM sizes {res.cluster_id_gmm.value_counts().to_dict()}; "
                 f"HDBSCAN clusters = {n_h} (min_cluster_size {mcs}), noise fraction "
                 f"{(res.cluster_id_hdbscan == -1).mean():.1%}")
    return {"best_k": best_k, "hdbscan_clusters": n_h}


def _plot(bics, best, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ks = sorted(bics)
    ax.plot(ks, [bics[k] for k in ks], "o-")
    ax.axvline(best, color="r", ls="--", label=f"best k = {best}")
    ax.set_xlabel("number of GMM components k")
    ax.set_ylabel("BIC (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bic.png", dpi=110)
    plt.close(fig)
