"""Stage 2d - Semantic audit: naming the clusters from INDEPENDENT evidence.

Purpose : turn abstract GMM cluster IDs into Star / Galaxy / Noise, and
          measure how good the learned segregation is against Gaia.
Method  : three evidence signals per object, none of them used to cluster:
            Star   - Gaia DR3 match (Stage 2c), OR on the stellar locus of the
                   size-magnitude plane: point-like (|FWHM / stellar locus - 1|
                   <= baseline.point_source_tolerance) with SNR >=
                   baseline.star_locus_min_snr, or saturated, AND confirmed in a
                   second band (a cosmic ray is as compact as a star but
                   single-band). Below that S/N the point-like band fills up with
                   compact distant galaxies that size cannot separate from stars
                   (check size_magnitude.png). Deep fields hold only a handful of
                   Gaia stars, too few to name a cluster alone;
            Galaxy - position matched to the Galaxy Zoo: Hubble catalogue, OR
                     resolved (FWHM / PSF > 1 + point_source_tolerance), SNR >= 5
                     and confirmed in a second band: GZH stops near mag 23.5, and
                     without this a cluster of fainter galaxies would carry no
                     galaxy evidence and be named Noise;
            Noise  - SNR < 3, a profile sharper than the PSF, or a single-band
                     detection (utils.noise_evidence).
          Each signal's per-cluster rate - measured on TRAIN-split members
          only - is divided by its maximum over the clusters (so a rare signal
          like Gaia is not swamped) and the strongest one names the cluster;
          several clusters may share a name.
          A cluster with no evidence takes the majority of the baseline rule.
          Gaia confirms stars but cannot reject them (it stops at G~21), so
          precision/recall are reported for STARS ONLY, inside Gaia's magnitude
          range, on HELD-OUT (non-train) objects, so naming and scoring never
          use the same Gaia matches; the baseline is scored WITHOUT its own Gaia leg.
          25-object montages per cluster support a visual check.
Output  : evidence.csv, cluster_names.json, semantic_audit.json, montage_cluster_*.png,
          size_magnitude.png (check the stellar locus before trusting the names)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..catalogs import load_gzh
from ..context import Ctx, load_json, load_objects, save_json
from ..features import load_cutouts
from ..utils import as_bool, asinh_normalise, fmt_offsets, match_with_offset, noise_evidence

NAMES = ("Star", "Galaxy", "Noise")


def star_metrics(pred, truth, in_range):
    p = (np.asarray(pred) == "Star") & in_range
    t = np.asarray(truth, bool) & in_range
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3), "n_gaia_stars": int(t.sum())}


def run(ctx: Ctx):
    cfg = ctx.cfg["semantic_audit"]
    out = ctx.stage_dir("2d")
    df = load_objects(ctx, ["sources", "pass1", "clusters", "baseline", "gaia"])
    metas = list(load_json(ctx.require("1a", "manifest.json")).values())

    gzh = load_gzh(ctx, metas, required=False)
    if gzh is not None and len(gzh):
        _, _, hit, off = match_with_offset(df.ra, df.dec, gzh.ra, gzh.dec, cfg["gzh_radius_arcsec"],
                                           estimate=ctx.cfg["catalogs"]["gzh"].get("estimate_offset", True),
                                           groups=df.pointing.to_numpy())
        ctx.log.info(f"[Stage 2d] GZH positions: {int(hit.sum())} matches (offset per pointing removed: "
                     f"{fmt_offsets(off)})")
    else:
        hit = np.zeros(len(df), bool)
        ctx.log.info("[Stage 2d] no GZH table - galaxy evidence falls back to the baseline rule")
    gaia = df.gaia_hit.astype(str).str.lower().eq("true").to_numpy()
    noise = noise_evidence(df, ctx.cfg["baseline"])
    locus_star = stellar_locus(df, ctx.cfg["baseline"])
    star_ev = gaia | locus_star
    resolved_real = resolved_two_band(df, ctx.cfg["baseline"])
    gal_ev = hit | resolved_real
    ctx.log.info(f"[Stage 2d] galaxy evidence: {int(hit.sum())} GZH + {int((resolved_real & ~hit).sum())} resolved and "
                 "confirmed in a second band")
    ctx.log.info(f"[Stage 2d] star evidence: {int(gaia.sum())} Gaia + {int((locus_star & ~gaia).sum())} on the stellar "
                 f"locus (point-like or saturated, confirmed in a second band)")
    ev = pd.DataFrame({"cluster": df.cluster_id_gmm, "Star": star_ev, "Galaxy": gal_ev, "Noise": noise})
    # Clusters are NAMED from TRAIN-split evidence only and SCORED on held-out objects: naming and
    # scoring on the same Gaia matches would be circular, and test objects must not shape the labels.
    tr = (df.split_tag == "train").to_numpy()
    rate = ev[tr].groupby("cluster")[list(NAMES)].mean()
    rate["n_train"] = ev[tr].groupby("cluster").size()
    no_train = sorted(set(ev.cluster) - set(rate.index))
    if no_train:                                   # a cluster with no train member: use all its members
        extra = ev[ev.cluster.isin(no_train)].groupby("cluster")[list(NAMES)].mean()
        extra["n_train"] = 0
        rate = pd.concat([rate, extra])
    rate["n"] = ev.groupby("cluster").size()
    score = rate[list(NAMES)] / rate[list(NAMES)].max().replace(0, 1)
    # a signal can only name a cluster if enough TRAIN members carry it (3 Gaia stars in a cluster of 400 are
    # not evidence that the cluster is stars - the max-normalisation would otherwise inflate a rare signal)
    min_n = int(cfg.get("min_evidence_members", 5))
    support = ev[tr].groupby("cluster")[list(NAMES)].sum().reindex(rate.index).fillna(0)
    score = score.where(support >= min_n, 0.0)
    names = {}
    for c, row in score.iterrows():
        if row.max() > 0:
            names[int(c)] = NAMES[int(np.argmax(row.values))]
        else:
            names[int(c)] = df.loc[df.cluster_id_gmm == c, "baseline_semantic_nogaia"].mode().iloc[0]
    rate["semantic"] = pd.Series(names)
    semantic = df.cluster_id_gmm.map(names)

    in_range = (df.mag < cfg["gaia_mag_limit"]).to_numpy()
    held = in_range & ~tr
    metrics = {"learned_star_vs_gaia": star_metrics(semantic, gaia, held),
               "baseline_star_vs_gaia": star_metrics(df.baseline_semantic_nogaia, gaia, held),
               "scored_on": "held-out objects (not train), detection-band mag < gaia_mag_limit",
               "learned_star_vs_gaia_all_objects": star_metrics(semantic, gaia, in_range),
               "baseline_star_vs_gaia_all_objects": star_metrics(df.baseline_semantic_nogaia, gaia, in_range),
               "learned_galaxy_recall_vs_gzh": round(float((semantic[hit & ~tr] == "Galaxy").mean()), 3)
               if (hit & ~tr).any() else None,
               "baseline_galaxy_recall_vs_gzh": round(float((df.baseline_semantic_nogaia[hit & ~tr] == "Galaxy").mean()), 3)
               if (hit & ~tr).any() else None,
               "cluster_names": {str(k): v for k, v in names.items()},
               "cluster_evidence": rate.round(3).reset_index().to_dict(orient="records")}
    metrics["clustering_validation"] = clustering_validation(df, ev, tr, names)
    ctx.log.info(f"[Stage 2d] external validation of the clusterings (held-out objects with exactly one evidence "
                 f"signal): {metrics['clustering_validation']['summary']}")
    l, b = metrics["learned_star_vs_gaia"], metrics["baseline_star_vs_gaia"]
    metrics["objective2_minimum_met"] = bool(l["n_gaia_stars"] > 0 and l["f1"] > b["f1"])
    save_json(metrics, out / "semantic_audit.json")
    save_json({str(k): v for k, v in names.items()}, out / "cluster_names.json")
    pd.DataFrame({"object_id": df.object_id, "gzh_position_hit": hit, "noise_evidence": noise,
                  "stellar_locus": locus_star, "cluster_semantic_2d": semantic}).to_csv(out / "evidence.csv", index=False)
    _montages(ctx, df, names, cfg["montage_n"], out)
    _size_magnitude(df, gaia, locus_star, noise, semantic, ctx.cfg["baseline"], out)
    ctx.log.info("[Stage 2d] cluster evidence (share of members with each signal):\n" + rate.round(3).to_string())
    ctx.log.info(f"[Stage 2d] stars vs Gaia (held-out, in range, n={l['n_gaia_stars']}): learned P={l['precision']} "
                 f"R={l['recall']} F1={l['f1']} | baseline P={b['precision']} R={b['recall']} F1={b['f1']} | "
                 f"Objective-2 minimum met: {metrics['objective2_minimum_met']}")
    return {"learned_f1": l["f1"], "baseline_f1": b["f1"]}


def clustering_validation(df, ev, tr, gmm_names):
    """EXTERNAL validation of GMM and HDBSCAN against the independent evidence (Gaia / stellar locus, GZH /
    resolved-two-band, noise signals), on HELD-OUT objects that carry exactly ONE signal:
      purity  = share of objects whose own evidence equals their cluster's majority evidence (cluster majority
                taken from TRAIN members), compared with the trivial 'everything is the majority class' baseline;
      NMI / ARI between cluster label and evidence label (0 = no information, 1 = identical partitions).
    A clustering is only useful if its purity beats the majority baseline."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    E = ev[list(NAMES)].astype(bool)
    one = (E.sum(axis=1) == 1).to_numpy()
    own = E.idxmax(axis=1).to_numpy()
    held = ~tr & one
    res = {"n_heldout_single_evidence": int(held.sum()),
           "majority_class_share": round(float(pd.Series(own[held]).value_counts(normalize=True).iloc[0]), 3)
           if held.any() else None}
    cols = {"gmm": df.cluster_id_gmm.to_numpy()}
    if "cluster_id_hdbscan" in df and np.isfinite(df.cluster_id_hdbscan.to_numpy(float)).any():
        cols["hdbscan"] = df.cluster_id_hdbscan.to_numpy(float)
    for name, lab in cols.items():
        lab = np.asarray(lab)
        ok = held & (lab.astype(float) >= 0)                      # HDBSCAN noise (-1) is not a cluster
        if not ok.any():
            continue
        maj = pd.Series(own[tr & one]).groupby(lab[tr & one]).agg(lambda s: s.value_counts().index[0])
        pred = pd.Series(lab[ok]).map(maj).fillna(pd.Series(own[tr & one]).mode().iloc[0])  # no train member: majority
        res[name] = {"n_scored": int(ok.sum()), "n_clusters": int(len(set(lab[lab.astype(float) >= 0]))),
                     "purity": round(float((pred.to_numpy() == own[ok]).mean()), 3),
                     "nmi_vs_evidence": round(float(normalized_mutual_info_score(own[ok], lab[ok])), 3),
                     "ari_vs_evidence": round(float(adjusted_rand_score(own[ok], lab[ok])), 3)}
        res[name]["purity_gain_over_majority"] = round(res[name]["purity"] - res["majority_class_share"], 3)
    sem = df.cluster_id_gmm.map(gmm_names).to_numpy()
    for c in NAMES:                                                # recall of each named population
        m = held & (own == c)
        if m.any():
            res.setdefault("gmm_named_recall", {})[c] = round(float((sem[m] == c).mean()), 3)
    res["summary"] = {k: {kk: v[kk] for kk in ("purity", "purity_gain_over_majority", "nmi_vs_evidence")}
                      for k, v in res.items() if isinstance(v, dict) and "purity" in v}
    res["summary"]["majority_class_share"] = res["majority_class_share"]
    return res


def two_band_confirmed(df):
    return df.cr_candidate.map(lambda v: str(v).strip().lower() in ("false", "0", "0.0")).to_numpy() \
        if "cr_candidate" in df else np.ones(len(df), bool)


def resolved_two_band(df, cfg_b, min_snr=5.0):
    """A real extended source: broader than a point source, SNR >= 5, confirmed in a second band."""
    ratio = np.nan_to_num(df.fwhm_over_psf.to_numpy(float), nan=0)
    return (ratio > 1 + cfg_b["point_source_tolerance"]) & (df.snr.to_numpy(float) >= min_snr) & two_band_confirmed(df)


def stellar_locus(df, cfg_b):
    """Point-like (or saturated) AND confirmed in a second band AND bright enough (SNR >= star_locus_min_snr)
    that size separates a star from a compact galaxy: the stellar sequence of the size-magnitude plane."""
    min_snr = float(cfg_b.get("star_locus_min_snr", 150))
    ratio = df.fwhm_over_psf.to_numpy(float)
    two_band = df.cr_candidate.map(lambda v: str(v).strip().lower() in ("false", "0", "0.0")).to_numpy() \
        if "cr_candidate" in df else np.ones(len(df), bool)
    bright = df.snr.to_numpy(float) >= min_snr
    point = np.abs(ratio - 1.0) <= cfg_b["point_source_tolerance"]
    sat = as_bool(df.saturated).to_numpy() if "saturated" in df else np.zeros(len(df), bool)
    return ((point & bright) | sat) & two_band


def _size_magnitude(df, gaia, locus, noise, semantic, cfg_b, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = df.fwhm_over_psf.to_numpy(float)
    m = df.mag.to_numpy(float)
    for a, title, groups in (
            (ax[0], "evidence", [(noise, "noise evidence", "0.75", 3), (~noise & ~locus & ~gaia, "other", "tab:blue", 4),
                                 (locus & ~gaia, "stellar locus", "tab:orange", 10), (gaia, "Gaia", "red", 30)]),
            (ax[1], "cluster names (Stage 2d)", [(np.asarray(semantic) == n, n, c, 4) for n, c in
                                                 (("Noise", "0.75"), ("Galaxy", "tab:blue"), ("Star", "tab:orange"))])):
        for mask, lab, col, s in groups:
            mask = np.asarray(mask, bool)
            a.scatter(m[mask], y[mask], s=s, c=col, label=f"{lab} ({int(mask.sum())})", edgecolors="none", alpha=0.7)
        a.axhspan(1 - cfg_b["point_source_tolerance"], 1 + cfg_b["point_source_tolerance"], color="k", alpha=0.08)
        lim = np.nanmedian(m[np.abs(df.snr.to_numpy(float) / cfg_b.get("star_locus_min_snr", 150) - 1) < 0.1])
        if np.isfinite(lim):
            a.axvline(lim, color="k", ls=":", lw=1)
            a.text(lim, a.get_ylim()[1] if a.get_yscale() == "linear" else 8, f" SNR {cfg_b.get('star_locus_min_snr', 150):g}",
                   fontsize=7, va="top")
        a.set_yscale("log")
        a.set_xlabel("detection-band AB magnitude")
        a.set_title(title)
        a.legend(fontsize=7, markerscale=1.5)
    ax[0].set_ylabel("FWHM / stellar-locus FWHM")
    fig.suptitle("Stage 2d: size-magnitude plane (grey band = point-like); stars sit on a horizontal sequence at 1")
    fig.tight_layout()
    fig.savefig(out / "size_magnitude.png", dpi=110)
    plt.close(fig)


def _montages(ctx, df, names, n, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cut = load_cutouts(ctx)
    rng = np.random.default_rng(ctx.seed)
    for c in sorted(df.cluster_id_gmm.unique()):
        members = df[df.cluster_id_gmm == c]
        pick = members.sample(min(n, len(members)), random_state=int(rng.integers(1e6)))
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), pick.cutout_index):
            ax.imshow(asinh_normalise(np.asarray(cut[i], np.float32).sum(0)), origin="lower", cmap="gray")
        fig.suptitle(f"GMM cluster {c} -> {names[int(c)]}  (n = {len(members)})")
        fig.tight_layout()
        fig.savefig(out / f"montage_cluster_{c}.png", dpi=80)
        plt.close(fig)
