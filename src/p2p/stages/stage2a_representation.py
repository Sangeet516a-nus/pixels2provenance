"""Stage 2a - Representation learning and feature fusion.

Purpose : one feature space that captures both pixel structure and physics.
Method  : a convolutional autoencoder (one input channel per band) compresses
          each asinh-scaled 128x128 cutout to a 64-d latent; PCA keeps the
          leading components (up to n_pca, dropping any below pca_min_variance;
          scaled as one block, not whitened). The classical Pass-1 features are median-imputed,
          signed-log transformed and RobustScaled. The two blocks are weighted
          (spatial_weight / 1 - spatial_weight) and concatenated.
          CAE, PCA, imputer and scaler are fitted on the TRAIN split only and
          applied with transform() to everything else (no leakage).
Output  : rep_cae.pt, latent.npy, hybrid_features.npy (row = sources.csv row), pca_scree.png,
          fusion.joblib, recon_error.npy
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..context import Ctx, load_objects, save_json
from ..features import TabularPrep, load_cutouts
from ..nets import ConvAutoencoder, ae_errors_and_latents, train_autoencoder
from ..utils import asinh_stack


def encode_all(model, cutouts, device, batch=256):
    errs, zs = [], []
    for i in range(0, len(cutouts), batch):
        X = asinh_stack(np.asarray(cutouts[i:i + batch], np.float32))
        e, z = ae_errors_and_latents(model, X, device)
        errs.append(e)
        zs.append(z)
    return np.concatenate(errs), np.concatenate(zs)


def run(ctx: Ctx):
    cfg = ctx.cfg["representation"]
    out = ctx.stage_dir("2a")
    df = load_objects(ctx, ["sources", "pass1"])
    cut = load_cutouts(ctx)
    train = df[df.split_tag == "train"]
    if len(train) < 20:
        raise RuntimeError(f"Only {len(train)} train-split sources - check Stage 0 regions / collar size.")
    device = ctx.device()
    X = asinh_stack(np.asarray(cut[train.cutout_index.to_numpy()], np.float32))
    ctx.log.info(f"[Stage 2a] training representation CAE ({cut.shape[1]} bands -> {cfg['latent_dim']}-d) "
                 f"on {len(X)} train-split cutouts ({device})")
    model = ConvAutoencoder(in_ch=cut.shape[1], latent_dim=cfg["latent_dim"], size=cut.shape[-1])
    # train for a minimum number of optimisation STEPS, not just epochs: with few training
    # cutouts, 15 epochs is only a few dozen steps and the latent collapses to one direction
    steps_per_epoch = int(np.ceil(len(X) / cfg["batch_size"]))
    epochs = max(int(cfg["epochs"]), int(np.ceil(cfg.get("min_steps", 0) / steps_per_epoch)))
    ctx.log.info(f"[Stage 2a] {epochs} epochs x {steps_per_epoch} steps = {epochs * steps_per_epoch} optimisation steps")
    train_autoencoder(model, X, epochs, cfg["batch_size"], device, log=ctx.log.info, tag="Rep-CAE")
    torch.save(model.state_dict(), out / "rep_cae.pt")

    errs, latent = encode_all(model, cut, device)
    rows = df.cutout_index.to_numpy()
    errs, latent = errs[rows], latent[rows]
    is_train = (df.split_tag == "train").to_numpy()
    # scree on the TRAIN latents; keep the leading components up to n_pca, but drop any that
    # explain less than pca_min_variance (a ~0 % component only adds noise to the clustering)
    scree = PCA(random_state=ctx.seed).fit(latent[is_train]).explained_variance_ratio_
    if scree[0] > 0.9:
        ctx.log.info(f"[Stage 2a] WARNING: PC1 explains {scree[0]:.1%} of the latent variance - the CAE latent has "
                     "collapsed to one direction (under-trained). Raise representation.min_steps.")
    min_var = float(cfg.get("pca_min_variance", 0.0))
    k = int(min(cfg["n_pca"], latent.shape[1]))
    k_used = max(2, min(k, int(np.sum(scree[:k] >= min_var))))
    if k_used < k:
        ctx.log.info(f"[Stage 2a] PC{k_used + 1}..PC{k} explain < {min_var:.0%} each "
                     f"({np.round(scree[k_used:k] * 100, 2).tolist()} %) -> keeping {k_used} components")
    pca = PCA(n_components=k_used, random_state=ctx.seed).fit(latent[is_train])
    Z = pca.transform(latent)
    _scree_plot(scree, k_used, out)
    z_scale = float(Z[is_train].std()) or 1.0          # one scale for the whole image block (not whitened:
    Z = Z / z_scale                                    # whitening gives noise components equal weight)
    prep = TabularPrep(cfg["tabular_features"]).fit(df[is_train])
    w = cfg["spatial_weight"]
    hybrid = np.hstack([Z * w, prep.transform(df) * (1 - w)]).astype(np.float32)

    np.save(out / "latent.npy", latent)
    np.save(out / "recon_error.npy", errs)
    np.save(out / "hybrid_features.npy", hybrid)
    np.save(out / "object_id.npy", df.object_id.to_numpy())
    joblib.dump({"pca": pca, "z_scale": z_scale, "prep": prep, "spatial_weight": w}, out / "fusion.joblib")
    save_json({"n_objects": len(df), "n_train": int(is_train.sum()), "hybrid_dim": hybrid.shape[1],
               "tabular_kept": prep.kept, "n_pca_used": k_used,
               "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
               "scree_first10_percent": np.round(scree[:10] * 100, 2).tolist()},
              out / "summary.json")
    ctx.log.info(f"[Stage 2a] hybrid matrix {hybrid.shape} | tabular kept {prep.kept} | "
                 f"PCA keeps {pca.explained_variance_ratio_.sum():.1%} of latent variance")
    return {"hybrid_dim": int(hybrid.shape[1])}


def _scree_plot(ratio, k_used, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(len(ratio), 15)
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(x, ratio[:n] * 100, color=["#2E5496" if i < k_used else "#BBBBBB" for i in range(n)])
    ax.plot(x, np.cumsum(ratio[:n]) * 100, "o-", color="#E69F00", ms=3, label="cumulative")
    ax.set_xticks(x)
    ax.set_xlabel("principal component of the CAE latent")
    ax.set_ylabel("explained variance [%]")
    ax.set_title(f"Stage 2a scree (train split) - {k_used} components kept")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pca_scree.png", dpi=110)
    plt.close(fig)
