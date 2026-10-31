"""Stage 2b - Unsupervised clustering (GMM + HDBSCAN), fitted on the TRAIN split only.

GMM     : full covariance (the features are correlated; a diagonal mixture never reaches a BIC minimum), k from
          the minimum BIC over [k_min, k_max] (BIC elbow if the minimum is on the boundary). Checks: ICL, AIC,
          held-out (dev) log-likelihood, BIC minimum with other seeds, bootstrap stability (ARI).
HDBSCAN : density-based; best DBCV over a small min_cluster_size grid (fits calling > hdbscan_max_noise of the
          sources noise are rejected). Every object gets the GLOSH outlier score; density_outlier = HDBSCAN noise
          or GLOSH above the train hdbscan_outlier_percentile (used by Stage 2e).
Other objects are assigned with GMM.predict / hdbscan.approximate_predict (never refitted). External validation
against Gaia / GZH / noise evidence is in Stage 2d (clustering_validation).
Output  : clusters.csv, gmm.joblib, hdbscan.joblib, gmm_k_selection.png / .csv, summary.json (incl. evaluation)
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from ..context import Ctx, save_json

try:
    import hdbscan as _hdb
except Exception:
    _hdb = None


def _fit_gmm(X, k, cov, seed, reg0, n_init=3):
    """GMM fit; reg_covar is raised x10 (up to 1e-1) if a component collapses onto identical points."""
    reg = reg0
    while reg <= 1e-1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return GaussianMixture(n_components=k, covariance_type=cov, random_state=seed, n_init=n_init,
                                       reg_covar=reg, max_iter=300).fit(X), reg
        except ValueError:
            reg *= 10
    return None, reg


def icl(g, X):
    """Integrated Completed Likelihood (lower is better, same scale as sklearn's BIC)."""
    t = g.predict_proba(X)
    return float(g.bic(X) + 2 * -np.sum(t * np.log(np.clip(t, 1e-300, 1))))


def knee(ks, vals):
    """Kneedle elbow: the point of the (normalised) decreasing curve farthest below the chord first -> last."""
    ks, v = np.asarray(ks, float), np.asarray(vals, float)
    if len(ks) < 3:
        return int(ks[int(np.argmin(v))])
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (v - v.min()) / max(v.max() - v.min(), 1e-12)
    chord = y[0] + (y[-1] - y[0]) * x
    return int(ks[int(np.argmax(chord - y))])


def run(ctx: Ctx):
    cfg = ctx.cfg["clustering"]
    out = ctx.stage_dir("2b")
    X = np.load(ctx.require("2a", "hybrid_features.npy"))
    oid = np.load(ctx.require("2a", "object_id.npy"))
    src = pd.read_csv(ctx.require("1c", "sources.csv")).set_index("object_id").loc[oid]
    tr = (src.split_tag == "train").to_numpy()
    dev = (src.split_tag == "dev").to_numpy()
    held = ~tr

    # float64: in float32 the variance of a tight component can round to <= 0 ("ill-defined covariance")
    X = X.astype(np.float64)
    Xtr = X[tr]
    f = pd.read_csv(ctx.require("1e", "features_pass1.csv"), usecols=["object_id", "flag"]).set_index("object_id")
    n_faint = int((f.reindex(oid)["flag"].to_numpy()[tr] == 4).sum())
    ctx.log.info(f"[Stage 2b] {n_faint} of {int(tr.sum())} train sources are too faint for morphology "
                 "(their shape features are median-filled, so they sit on top of each other)")
    cov = cfg.get("covariance_type", "full")
    reg0 = float(cfg.get("reg_covar", 1e-3))
    n_init = int(cfg.get("n_init", 3))

    # ---------------- GMM: sweep k ----------------
    rows, models = [], {}
    for k in range(int(cfg["k_min"]), int(cfg["k_max"]) + 1):
        g, reg = _fit_gmm(Xtr, k, cov, ctx.seed, reg0, n_init)
        if g is None:
            ctx.log.info(f"  [GMM] k={k}  failed (collapsed component even with reg_covar 1e-1) - skipped")
            continue
        models[k] = g
        r = {"k": k, "bic": g.bic(Xtr), "aic": g.aic(Xtr), "icl": icl(g, Xtr),
             "heldout_loglik": float(g.score(X[dev])) if dev.any() else np.nan, "reg_covar": reg}
        rows.append(r)
        ctx.log.info(f"  [GMM {cov}] k={k:2d}  BIC={r['bic']:.1f}  ICL={r['icl']:.1f}  AIC={r['aic']:.1f}  "
                     f"dev log-lik/object={r['heldout_loglik']:.3f}" + (f"  (reg_covar {reg:g})" if reg > reg0 else ""))
    if not rows:
        raise RuntimeError("Every GMM fit collapsed. Lower clustering.k_max or raise clustering.reg_covar.")
    sel = pd.DataFrame(rows)
    k_bic = int(sel.k[sel.bic.idxmin()])
    k_icl = int(sel.k[sel.icl.idxmin()])
    k_knee = knee(sel.k, sel.bic)
    rule = "minimum BIC"
    best_k = k_bic
    if k_bic == int(sel.k.max()):
        best_k, rule = k_knee, "elbow (knee) of BIC - the BIC minimum is at the upper boundary of the range"
        ctx.log.info(f"[Stage 2b] WARNING: BIC still falling at k_max={k_bic}; using the BIC elbow k={k_knee}. "
                     "Raise clustering.k_max or check the features (correlated / tied values).")

    # robustness 1: does the minimum move with other seeds?
    seed_check = {}
    ks_near = [k for k in range(max(int(cfg["k_min"]), best_k - 2), min(int(cfg["k_max"]), best_k + 2) + 1)]
    for s in cfg.get("robustness_seeds", [7, 1]):
        b = {}
        for k in ks_near:
            g, _ = _fit_gmm(Xtr, k, cov, int(s), reg0, n_init)
            if g is not None:
                b[k] = g.bic(Xtr)
        seed_check[str(s)] = int(min(b, key=b.get)) if b else None
    # robustness 2: bootstrap stability of the chosen model (ARI of held-out labels)
    gmm = models[best_k]
    lab_all = gmm.predict(X)
    rng = np.random.default_rng(ctx.seed)
    aris = []
    for b in range(int(cfg.get("bootstrap_refits", 3))):
        bi = rng.choice(np.where(tr)[0], int(tr.sum()), replace=True)
        g2, _ = _fit_gmm(X[bi], best_k, cov, 100 + b, reg0, n_init)
        if g2 is not None:
            aris.append(float(adjusted_rand_score(lab_all[held], g2.predict(X[held]))))
    ctx.log.info(f"[Stage 2b] GMM k = {best_k} ({rule}); ICL minimum k = {k_icl}; BIC elbow k = {k_knee}; "
                 f"BIC minimum with other seeds: {seed_check}; bootstrap stability ARI {np.round(aris, 3).tolist()}")

    probs = gmm.predict_proba(X)
    res = pd.DataFrame({"object_id": oid, "cluster_id_gmm": probs.argmax(1)})
    for j in range(best_k):
        res[f"gmm_p{j}"] = probs[:, j].round(5)
    joblib.dump(gmm, out / "gmm.joblib")

    # comparison curve: the diagonal-covariance BIC (why the covariance type matters)
    diag_curve = {}
    if cfg.get("compare_diag", True) and cov != "diag":
        for k in sel.k:
            g, _ = _fit_gmm(Xtr, int(k), "diag", ctx.seed, reg0, n_init=1)
            if g is not None:
                diag_curve[int(k)] = float(g.bic(Xtr))
    sel["bic_diag_covariance"] = sel.k.map(diag_curve)
    sel.to_csv(out / "gmm_k_selection.csv", index=False)

    # ---------------- HDBSCAN ----------------
    mcs = cfg["hdbscan_min_cluster_size"]
    mcs = max(10, len(Xtr) // 50) if mcs == "auto" else int(mcs)
    hdb_status, hdb_trials, n_h, hdb_internal = "not run", [], 0, {}
    max_noise = float(cfg.get("hdbscan_max_noise", 0.5))
    if _hdb is not None:
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
            noise = float((h.labels_ == -1).mean())
            hdb_trials.append({"min_cluster_size": m, "clusters": n_c, "noise_fraction": round(noise, 4),
                               "dbcv": None if not np.isfinite(dbcv) else round(dbcv, 4),
                               "largest_cluster_fraction": round(float(np.max(np.bincount(h.labels_[h.labels_ >= 0]))
                                                                       / len(h.labels_)), 4) if n_c else None})
            ctx.log.info(f"  [HDBSCAN] min_cluster_size={m}: {n_c} clusters, noise {noise:.1%}, DBCV {hdb_trials[-1]['dbcv']}, "
                         f"largest cluster {hdb_trials[-1]['largest_cluster_fraction']}")
            if n_c >= 2 and noise <= max_noise and (best is None or np.nan_to_num(dbcv, nan=-1) > best[2]):
                best = (h, m, np.nan_to_num(dbcv, nan=-1))
        if best is not None:
            h, mcs = best[0], best[1]
            lab, strength = _hdb.approximate_predict(h, X)
            glosh = np.asarray(_hdb.approximate_predict_scores(h, X), float)
            lab[tr], glosh[tr] = h.labels_, h.outlier_scores_          # training points keep their fitted values
            glosh = np.nan_to_num(glosh, nan=1.0)
            thr = float(np.percentile(glosh[tr], float(cfg.get("hdbscan_outlier_percentile", 95))))
            res["cluster_id_hdbscan"] = lab
            res["hdbscan_strength"] = np.round(strength, 4)
            res["hdbscan_glosh"] = np.round(glosh, 4)
            res["density_outlier"] = (lab == -1) | (glosh > thr)
            joblib.dump(h, out / "hdbscan.joblib")
            n_h = len(set(h.labels_) - {-1})
            sizes = np.bincount(h.labels_[h.labels_ >= 0])
            hdb_internal = {"dbcv": round(float(best[2]), 4), "clusters": n_h,
                            "noise_fraction_train": round(float((h.labels_ == -1).mean()), 4),
                            "largest_cluster_fraction_train": round(float(sizes.max() / len(h.labels_)), 4),
                            "glosh_threshold": round(thr, 4),
                            "density_outlier_fraction": round(float(res.density_outlier.mean()), 4),
                            "ari_vs_gmm_train": round(float(adjusted_rand_score(lab_all[tr], h.labels_)), 4),
                            "nmi_vs_gmm_train": round(float(normalized_mutual_info_score(lab_all[tr], h.labels_)), 4)}
            if hdb_internal["largest_cluster_fraction_train"] > 0.8:
                ctx.log.info(f"[Stage 2b] NOTE: one HDBSCAN cluster holds {hdb_internal['largest_cluster_fraction_train']:.0%} "
                             "of the train sources - the feature space is one density continuum with small islands, "
                             "not separate populations; HDBSCAN is used for density outliers (GLOSH), the GMM for the populations.")
            hdb_status = "ok"
        else:
            res["cluster_id_hdbscan"] = np.nan
            hdb_status = "failed"
            ctx.log.info(f"[Stage 2b] HDBSCAN FAILED: no min_cluster_size in {grid} gives >= 2 clusters with noise <= "
                         f"{max_noise:.0%}. cluster_id_hdbscan is left empty and Stage 2e gating runs without it.")
    else:
        res["cluster_id_hdbscan"] = np.nan
        ctx.log.info("[Stage 2b] hdbscan package missing (pip install hdbscan) - HDBSCAN skipped")
    res.to_csv(out / "clusters.csv", index=False)
    evaluation = _evaluate(ctx, sel, best_k, seed_check, aris, hdb_internal, res, src.split_tag.to_numpy())
    summary = {"covariance_type": cov, "k_range": [int(cfg["k_min"]), int(cfg["k_max"])], "best_k": best_k,
               "k_rule": rule, "k_min_bic": k_bic, "k_min_icl": k_icl, "k_bic_elbow": k_knee,
               "k_min_bic_other_seeds": seed_check, "bootstrap_stability_ari": [round(a, 3) for a in aris],
               "bic": {int(r.k): float(r.bic) for r in sel.itertuples()},
               "icl": {int(r.k): float(r.icl) for r in sel.itertuples()},
               "heldout_loglik": {int(r.k): float(r.heldout_loglik) for r in sel.itertuples()},
               "bic_diag_covariance": diag_curve,
               "hdbscan_status": hdb_status, "hdbscan_trials": hdb_trials, "hdbscan_min_cluster_size": mcs,
               "hdbscan_clusters": n_h, "hdbscan_internal": hdb_internal,
               "gmm_sizes": {int(k): int(v) for k, v in res.cluster_id_gmm.value_counts().items()},
               "evaluation": evaluation}
    save_json(summary, out / "summary.json")
    _plot(sel, best_k, k_icl, k_knee, rule, cov, out)
    ctx.log.info(f"[Stage 2b] GMM k = {best_k}; sizes {summary['gmm_sizes']}; HDBSCAN clusters = {n_h} "
                 f"(min_cluster_size {mcs}); density outliers {float(res.get('density_outlier', pd.Series([0])).mean()):.1%}")
    return {"best_k": best_k, "hdbscan_clusters": n_h}


def _evaluate(ctx, sel, best_k, seed_check, aris, hdb, res, split):
    """Internal checks of both clusterings, logged in plain words (external check vs evidence: Stage 2d)."""
    b = sel.set_index("k").bic
    others = b.drop(best_k)
    runner = int(others.idxmin()) if len(others) else None
    ari = float(np.mean(aris)) if aris else np.nan
    seeds_agree = all(v == best_k for v in seed_check.values())
    ev = {"gmm": {"k": best_k, "bic_margin_to_next_best_k": None if runner is None else
                  {"k": runner, "delta_bic": round(float(others.min() - b[best_k]), 1)},
                  "heldout_loglik_best_k": int(sel.k[sel.heldout_loglik.idxmax()]),
                  "seeds_agree_on_k": seeds_agree, "bootstrap_ari_mean": round(ari, 3),
                  "stability": "stable" if ari >= 0.8 else "moderately stable" if ari >= 0.5 else "unstable"}}
    g = ev["gmm"]
    ctx.log.info(f"[Stage 2b] GMM check: k = {best_k}; next-best k = {runner} (BIC +{g['bic_margin_to_next_best_k']['delta_bic'] if runner else '-'}); "
                 f"dev log-likelihood peaks at k = {g['heldout_loglik_best_k']}; other seeds "
                 f"{'agree' if seeds_agree else 'disagree: ' + str(seed_check)}; bootstrap ARI {g['bootstrap_ari_mean']} "
                 f"-> {g['stability']} (>= 0.8 stable, 0.5-0.8 moderate). The exact k matters little: Stage 2d "
                 "names clusters by evidence and several clusters can share a name.")
    if hdb:
        rate = pd.Series(res.density_outlier.to_numpy()).groupby(split).mean().round(3).to_dict()
        ev["hdbscan"] = {"dbcv": hdb["dbcv"], "clusters": hdb["clusters"],
                         "largest_cluster_fraction_train": hdb["largest_cluster_fraction_train"],
                         "density_outlier_rate_by_split": rate,
                         "structure": "none (one density continuum)" if hdb["largest_cluster_fraction_train"] > 0.8
                         or hdb["dbcv"] < 0.1 else "clusters found"}
        ctx.log.info(f"[Stage 2b] HDBSCAN check: DBCV {hdb['dbcv']} (-1..1; < 0.1 = no density structure), "
                     f"{hdb['clusters']} clusters, largest {hdb['largest_cluster_fraction_train']:.0%} of train -> "
                     f"{ev['hdbscan']['structure']}; used only for density outliers. Outlier rate by split: {rate} "
                     f"(train ~{100 - float(ctx.cfg['clustering'].get('hdbscan_outlier_percentile', 95)):g} % by "
                     "construction; buffer = collars and sources outside the three-band area)")
    return ev


def _plot(sel, best, k_icl, k_knee, rule, cov, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, aqua, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    ks = sel.k.to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
    ax = axs[0]
    ax.plot(ks, sel.bic / 1e3, "o-", color=blue, lw=2, ms=4, label="BIC")
    ax.plot(ks, sel.icl / 1e3, "s-", color=orange, lw=2, ms=4, label="ICL (BIC + assignment entropy)")
    ax.plot(ks, sel.aic / 1e3, "^-", color=aqua, lw=1.5, ms=4, label="AIC")
    ax.axvline(best, color=ink, ls="--", lw=1)
    ax.annotate(f"chosen k = {best}", (best, sel.bic.min() / 1e3), xytext=(14, -22), textcoords="offset points",
                fontsize=8.5, color=ink, arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8})
    ax.set_xlabel("number of GMM components k")
    ax.set_ylabel("criterion / 1000 (lower is better)")
    ax.set_title(f"Elbow / information criteria ({cov} covariance)\n{rule}; ICL min k = {k_icl}, BIC knee k = {k_knee}",
                 fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax = axs[1]
    ax.plot(ks, sel.heldout_loglik, "o-", color=blue, lw=2, ms=4)
    ax.axvline(best, color=ink, ls="--", lw=1)
    ax.set_xlabel("number of GMM components k")
    ax.set_ylabel("mean log-likelihood per object")
    ax.set_title("Held-out (dev region) log-likelihood\nhigher = generalises better", fontsize=9)
    ax = axs[2]
    if sel.bic_diag_covariance.notna().any():
        ax.plot(ks, sel.bic_diag_covariance / 1e3, "o-", color=muted, lw=1.5, ms=4, label="diagonal covariance")
        ax.plot(ks, sel.bic / 1e3, "o-", color=blue, lw=2, ms=4, label=f"{cov} covariance (used)")
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_title("BIC: diagonal vs full covariance\n(diagonal keeps falling - no minimum)", fontsize=9)
    else:
        ax.axis("off")
    ax.set_xlabel("number of GMM components k")
    ax.set_ylabel("BIC / 1000")
    from matplotlib.ticker import MaxNLocator
    for a in axs:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "gmm_k_selection.png", dpi=120, facecolor="white")
    plt.close(fig)
