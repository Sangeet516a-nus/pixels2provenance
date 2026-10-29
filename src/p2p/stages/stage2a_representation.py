"""Stage 2a - Representation learning and feature fusion.

Purpose : one feature space that holds both pixel structure and physics.
Method  : a convolutional autoencoder (CAE) encodes every cutout to a 32-d latent code. Input: each band on one
          noise scale per pointing (background-subtracted, / band sigma, asinh) plus a "data present" mask per
          band; loss: MSE weighted x target_weight on the source, 0 on cosmic-ray and no-data pixels; rotations /
          flips; early stopping on the dev split. PCA keeps the components above chance (Horn's parallel analysis).
          The tabular features are median-imputed, signed-log transformed and RobustScaled. Both blocks are
          weighted (spatial_weight / 1 - spatial_weight) and concatenated. Everything is fitted on the TRAIN split
          only (dev only decides when to stop) and applied to all other objects.
Output  : rep_cae.pt, latent.npy, recon_error.npy, hybrid_features.npy (row = sources.csv row), object_id.npy,
          fusion.joblib, pca_scree.png, cae_training.png / .json, cae_diagnostics.json, summary.json
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..context import Ctx, load_json, load_objects, save_json
from ..features import TabularPrep, load_cutouts
from ..nets import RepAutoencoder, rep_errors, train_rep_autoencoder
from ..representation import band_levels, band_sigmas, normalise_cutouts, pixel_weights, spearman


def run(ctx: Ctx):
    cfg = ctx.cfg["representation"]
    out = ctx.stage_dir("2a")
    df = load_objects(ctx, ["sources", "pass1"])
    cut = load_cutouts(ctx)
    is_train = (df.split_tag == "train").to_numpy()
    if is_train.sum() < 20:
        raise RuntimeError(f"Only {int(is_train.sum())} train-split sources - check Stage 0 regions / collar size.")
    latent, errs, diag = _train_cae(ctx, cfg, df, cut, is_train, ctx.device(), out)

    # ---- PCA of the latent (train split), scree + what each component tracks ----
    full = PCA(svd_solver="full", random_state=ctx.seed).fit(latent[is_train])
    scree = full.explained_variance_ratio_
    if scree[0] > 0.9:
        ctx.log.info(f"[Stage 2a] WARNING: PC1 explains {scree[0]:.1%} of the latent variance - the CAE latent has "
                     "collapsed to one direction (under-trained).")
    min_var = float(cfg.get("pca_min_variance", 0.0))
    pa = parallel_analysis(latent[is_train], n_iter=int(cfg.get("pca_parallel_iterations", 50)), seed=ctx.seed)
    diag["parallel_analysis"] = pa
    if str(cfg["n_pca"]).lower() == "auto":
        # Horn's parallel analysis: components with more variance than the same data with every column shuffled
        k = int(min(pa["n_components"], int(cfg.get("n_pca_max", 12)), latent.shape[1]))
        ctx.log.info(f"[Stage 2a] parallel analysis: {pa['n_components']} components above the 95th percentile of "
                     f"shuffled-data eigenvalues -> n_pca = {k} (cap n_pca_max {cfg.get('n_pca_max', 12)})")
    else:
        k = int(min(cfg["n_pca"], latent.shape[1]))
        ctx.log.info(f"[Stage 2a] n_pca fixed at {k} (parallel analysis would keep {pa['n_components']})")
    k_used = max(2, min(k, int(np.sum(scree[:k] >= min_var))))
    if k_used < k:
        ctx.log.info(f"[Stage 2a] PC{k_used + 1}..PC{k} explain < {min_var:.0%} each "
                     f"({np.round(scree[k_used:k] * 100, 2).tolist()} %) -> keeping {k_used} components")
    pca = PCA(n_components=k_used, svd_solver="full", random_state=ctx.seed).fit(latent[is_train])
    Z = pca.transform(latent)
    pca_all = full.transform(latent)                   # all components (same axes as pca)
    corr = _component_correlations(ctx, df, cut, pca_all[:, :8], cfg["tabular_features"])
    diag["component_correlations"] = corr
    diag["max_abs_rho_kept_pcs_vs_band_levels"] = round(float(np.nanmax(np.abs(
        np.array([[corr[f"PC{i + 1}"][c] for c in corr["band_level_columns"]] for i in range(k_used)])))), 3)
    _scree_plot(scree, k_used, corr, out, pa=pa)
    ctx.log.info(f"[Stage 2a] largest |Spearman rho| between a kept component and a band background level: "
                 f"{diag['max_abs_rho_kept_pcs_vs_band_levels']} (the original CAE: 0.98)")

    z_scale = float(Z[is_train].std()) or 1.0          # one scale for the image block (not whitened)
    Z = Z / z_scale
    prep = TabularPrep(cfg["tabular_features"]).fit(df[is_train])
    w = cfg["spatial_weight"]
    hybrid = np.hstack([Z * w, prep.transform(df) * (1 - w)]).astype(np.float32)
    diag["probe"] = _probe(df, Z, is_train)
    diag["probe_vs_n_components"] = {str(n): _probe(df, pca_all[:, :n], is_train)
                                     for n in sorted({2, 4, k_used, min(12, pca_all.shape[1]), pca_all.shape[1]})}
    ctx.log.info(f"[Stage 2a] linear probe vs number of components (information kept): {diag['probe_vs_n_components']}")
    ctx.log.info(f"[Stage 2a] linear probe on the kept components (train -> dev, balanced accuracy): {diag['probe']}")

    np.save(out / "latent.npy", latent)
    np.save(out / "recon_error.npy", errs)
    np.save(out / "hybrid_features.npy", hybrid)
    np.save(out / "object_id.npy", df.object_id.to_numpy())
    joblib.dump({"pca": pca, "z_scale": z_scale, "prep": prep, "spatial_weight": w,
                 "cae_version": 2, "band_sigma": diag.get("band_sigma")}, out / "fusion.joblib")
    save_json(diag, out / "cae_diagnostics.json")
    save_json({"n_objects": len(df), "n_train": int(is_train.sum()), "hybrid_dim": hybrid.shape[1],
               "tabular_kept": prep.kept, "cae_version": 2, "n_pca_used": k_used,
               "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
               "scree_first10_percent": np.round(scree[:10] * 100, 2).tolist(),
               "n_pca_rule": "parallel analysis (95th pct of shuffled eigenvalues)" if str(cfg["n_pca"]).lower() == "auto"
               else "fixed", "n_pca_parallel_analysis": pa["n_components"],
               "max_abs_rho_kept_pcs_vs_band_levels": diag["max_abs_rho_kept_pcs_vs_band_levels"],
               "recon_gain_vs_flat_dev": diag.get("recon_gain_vs_flat", {}).get("dev_median")},
              out / "summary.json")
    ctx.log.info(f"[Stage 2a] hybrid matrix {hybrid.shape} | tabular kept {prep.kept} | "
                 f"PCA keeps {pca.explained_variance_ratio_.sum():.1%} of latent variance")
    return {"hybrid_dim": int(hybrid.shape[1]), "n_pca": k_used,
            "max_rho_band_levels": diag["max_abs_rho_kept_pcs_vs_band_levels"]}


# ===================================================================== CAE
def _train_cae(ctx, cfg, df, cut, is_train, device, out):
    seg = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    rows = df.cutout_index.to_numpy()
    a, m = float(cfg.get("asinh_softening_sigma", 3.0)), float(cfg.get("asinh_max_sigma", 1000.0))
    # background sigma per band and pointing (image-noise statistic, no labels)
    sig_rows = np.zeros((len(df), cut.shape[1]), np.float32)
    band_sigma = {}
    for pid, sub in df.groupby("pointing"):
        s = band_sigmas(cut, seg, sub.cutout_index.to_numpy(), seed=ctx.seed)
        band_sigma[str(pid)] = [None if not np.isfinite(v) else round(float(v), 6) for v in s]
        sig_rows[(df.pointing == pid).to_numpy()] = s
    ctx.log.info(f"[Stage 2a] background sigma per band (data units) {band_sigma}")
    X, P = normalise_cutouts(cut, seg, rows, sig_rows, a, m)
    W = pixel_weights(np.asarray(seg)[rows], P, float(cfg.get("target_weight", 10.0)),
                      bool(cfg.get("ignore_cr_pixels", True)))
    missing = (~P.any(axis=(2, 3))).mean(axis=0)
    ctx.log.info(f"[Stage 2a] cutouts without data per band: {np.round(missing * 100, 1).tolist()} % "
                 f"(fed to the CAE as 'data present' channels)")

    tr = np.where(is_train)[0]
    va = np.where((df.split_tag == "dev").to_numpy())[0]
    if len(va) < 50:                                   # no usable dev region: hold out 10 % of train for stopping
        rng = np.random.default_rng(ctx.seed)
        va = rng.choice(tr, max(50, len(tr) // 10), replace=False)
        tr = np.setdiff1d(tr, va)
    torch.manual_seed(ctx.seed)
    np.random.seed(ctx.seed)
    model = RepAutoencoder(bands=cut.shape[1], latent_dim=int(cfg["latent_dim"]), size=cut.shape[-1],
                           channels=tuple(cfg.get("channels", [16, 32, 64, 64])),
                           mask_channels=bool(cfg.get("band_mask_channels", True)))
    ctx.log.info(f"[Stage 2a] training representation CAE v2 ({cut.shape[1]} bands + presence masks -> "
                 f"{cfg['latent_dim']}-d, {sum(p.numel() for p in model.parameters()):,} weights) on {len(tr)} "
                 f"train cutouts, validation {len(va)} ({device}); at most {cfg.get('max_epochs', 150)} epochs, "
                 f"early stopping after {cfg.get('patience', 10)} epochs without a {float(cfg.get('min_delta', 0.005)):.1%} improvement")
    ck, hj = out / "rep_cae.pt", out / "cae_training.json"
    if cfg.get("reuse_trained_cae", False) and ck.exists() and hj.exists():
        # re-run only the PCA / fusion with the CAE trained by an earlier run (no retraining)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.to(device)
        hist = load_json(hj)
        ctx.log.info(f"[Stage 2a] reuse_trained_cae: loaded {ctx.rel(ck)} (best epoch {hist['best_epoch']}) - not retrained")
    else:
        hist = train_rep_autoencoder(model, X, P, W, tr, va, device, max_epochs=int(cfg.get("max_epochs", 100)),
                                     min_epochs=int(cfg.get("min_epochs", 20)), patience=int(cfg.get("patience", 10)),
                                     batch_size=int(cfg["batch_size"]), lr=float(cfg.get("lr", 1e-3)),
                                     lr_patience=int(cfg.get("lr_patience", 4)), augment=bool(cfg.get("augment", True)),
                                     min_delta=float(cfg.get("min_delta", 0.005)), checkpoint=ck,
                                     log=ctx.log.info)              # best weights are written to rep_cae.pt
    ctx.log.info(f"[Stage 2a] best validation loss {hist['best_val']:.5f} at epoch {hist['best_epoch']} "
                 f"of {len(hist['val'])}")
    errs, latent = rep_errors(model, X, P, W, np.arange(len(df)), device)

    # reconstruction gain over a flat image (each band filled with its weighted mean)
    gains = {}
    for name, idx in (("train", tr), ("dev", va)):
        e_flat = _flat_error(X, P, W, idx)
        g = 1 - errs[idx] / np.maximum(e_flat, 1e-12)
        gains[f"{name}_median"] = round(float(np.median(g)), 3)
        gains[f"{name}_fraction_better"] = round(float((g > 0).mean()), 3)
    ctx.log.info(f"[Stage 2a] reconstruction gain over a flat per-band image: {gains}")
    _training_plot(hist, out)
    save_json(hist, out / "cae_training.json")
    diag = {"cae_version": 2, "band_sigma": band_sigma, "missing_band_fraction": np.round(missing, 4).tolist(),
            "epochs_run": len(hist["val"]), "best_epoch": hist["best_epoch"], "best_val_loss": hist["best_val"],
            "recon_gain_vs_flat": gains, "n_weights": int(sum(p.numel() for p in model.parameters()))}
    return latent, errs, diag


def _flat_error(X, P, W, idx, batch=256):
    out = []
    for s in range(0, len(idx), batch):
        b = np.sort(idx[s:s + batch])
        x, p, w = (np.asarray(A[b], np.float32) for A in (X, P, W))
        mean = (x * w).sum(axis=(2, 3), keepdims=True) / np.maximum(w.sum(axis=(2, 3), keepdims=True), 1e-6)
        e = (w * (x - mean) ** 2).mean(axis=(1, 2, 3))
        order = np.argsort(np.argsort(idx[s:s + batch]))       # back to the order of idx
        out.append(e[order])
    return np.concatenate(out)


# ===================================================================== diagnostics
def _component_correlations(ctx, df, cut, Zall, physical, n_sample=1500):
    """Spearman rho of each principal component with the band background levels (a component that tracks them
    encodes sky level, not structure) and with the tabular features, on a random sample of objects."""
    rng = np.random.default_rng(ctx.seed)
    s = np.sort(rng.choice(len(df), min(n_sample, len(df)), replace=False))
    lev = band_levels(cut, df.cutout_index.to_numpy()[s])
    bands = list(load_json(ctx.require("1a", "pointings.json"))["bands"])
    level_cols = [f"level_{b}" for b in bands][:lev.shape[1]]
    cols = {c: lev[:, i] for i, c in enumerate(level_cols)}
    physical = [c for c in physical if c in df]
    cols.update({c: df[c].to_numpy(float)[s] for c in physical})
    res = {"band_level_columns": level_cols, "physical_columns": physical}
    for i in range(Zall.shape[1]):
        res[f"PC{i + 1}"] = {c: round(spearman(Zall[s, i], v), 3) for c, v in cols.items()}
    return res


def _probe(df, Z, is_train):
    """How much the kept image components alone say about (a) cosmic-ray candidates and (b) point-like
    vs extended sources: logistic regression fitted on train, balanced accuracy on dev."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    dev = (df.split_tag == "dev").to_numpy()
    out = {}
    targets = {}
    if "cr_candidate" in df:
        targets["cosmic_ray_candidate"] = df.cr_candidate.astype(float).to_numpy()
    f = df.get("fwhm_over_psf")
    if f is not None:
        y = np.where(f < 1.15, 0.0, np.where(f > 1.4, 1.0, np.nan))
        y[(df.snr < 10).to_numpy()] = np.nan                   # size is only meaningful at S/N >= 10
        targets["extended_vs_pointlike"] = y
    for name, y in targets.items():
        tr, te = is_train & np.isfinite(y), dev & np.isfinite(y)
        if tr.sum() < 30 or te.sum() < 10 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z[tr], y[tr])
        out[name] = round(float(balanced_accuracy_score(y[te], clf.predict(Z[te]))), 3)
    return out


def parallel_analysis(Xtr, n_iter=50, q=95, seed=42, max_rows=5000):
    """Horn (1965) parallel analysis. Eigenvalues of the train latent vs the q-th percentile of eigenvalues of
    the same matrix with each column permuted independently (marginals kept, correlations destroyed).
    Returns the number of leading components above that null and both spectra (as variance fractions)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(Xtr, np.float64)
    if len(X) > max_rows:
        X = X[rng.choice(len(X), max_rows, replace=False)]
    X = X - X.mean(0)
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    null = np.array([np.linalg.eigvalsh(np.cov(np.column_stack([rng.permutation(c) for c in X.T]),
                                                rowvar=False))[::-1] for _ in range(n_iter)])
    thr = np.percentile(null, q, axis=0)
    above = ev > thr
    n = int(np.argmin(above)) if not above.all() else len(ev)
    tot = ev.sum()
    return {"n_components": n, "eigen_fraction": np.round(ev / tot, 5).tolist(),
            "null_q_fraction": np.round(thr / tot, 5).tolist(), "percentile": q, "iterations": n_iter}


def _scree_plot(ratio, k_used, corr, out, pa=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(len(ratio), 15)
    x = np.arange(1, n + 1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    ax.bar(x, ratio[:n] * 100, color=["#2E5496" if i < k_used else "#BBBBBB" for i in range(n)])
    ax.plot(x, np.cumsum(ratio[:n]) * 100, "o-", color="#E69F00", ms=3, label="cumulative")
    if pa is not None:
        ax.plot(x, np.asarray(pa["null_q_fraction"][:n]) * 100, "s--", color="#C00000", ms=3, lw=1,
                label=f"shuffled data, {pa['percentile']}th pct (parallel analysis: {pa['n_components']})")
    ax.set_xticks(x)
    ax.set_xlabel("principal component of the CAE latent")
    ax.set_ylabel("explained variance [%]")
    ax.set_title(f"Stage 2a scree (train split) - {k_used} kept")
    ax.legend(loc="center right", fontsize=7)
    cols = corr["band_level_columns"] + corr["physical_columns"]
    pcs = [k for k in corr if k.startswith("PC")]
    M = np.array([[abs(corr[p][c]) if np.isfinite(corr[p][c]) else np.nan for c in cols] for p in pcs])
    im = bx.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([c.replace("level_", "bg level ") for c in cols], rotation=60, ha="right", fontsize=7.5)
    bx.set_yticks(range(len(pcs)))
    bx.set_yticklabels([f"{p} ({ratio[i] * 100:.1f} %)" for i, p in enumerate(pcs)], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                bx.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.6 else "black")
    nb = len(corr["band_level_columns"])
    bx.axvline(nb - 0.5, color="#C00000", lw=1)
    bx.axhline(k_used - 0.5, color="#555555", lw=0.8, ls="--")
    bx.set_title("|Spearman rho| of each component with band background levels (left of red line)\n"
                 "and physical features; dashed line = last kept component", fontsize=8.5)
    fig.colorbar(im, ax=bx, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(out / "pca_scree.png", dpi=110)
    plt.close(fig)


def _training_plot(hist, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["train"]) + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ep, hist["train"], label="train (augmented)", color="#2E5496")
    ax.plot(ep, hist["val"], label="validation (dev region)", color="#E69F00")
    ax.axvline(hist["best_epoch"], color="0.5", ls="--", lw=0.8, label=f"best epoch {hist['best_epoch']}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted reconstruction MSE")
    ax.set_title("Stage 2a: representation CAE training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cae_training.png", dpi=110)
    plt.close(fig)
