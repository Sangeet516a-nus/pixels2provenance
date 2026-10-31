"""Stage 1b - Frame quality control with a convolutional autoencoder.

Purpose : flag detector artefacts (satellite trails, bad drizzle regions, chip
          edges, hot-pixel clusters) that survive drizzle combination.
Method  : 128x128 sub-tiles of every SCI image are asinh-normalised; an
          under-complete CAE is trained on a random sample of well-covered
          tiles (artefacts are rare, so the model learns 'normal sky'); every
          tile is then scored with a 64-px stride (50% overlap, so an artefact
          on a tile edge is not split). The score is the reconstruction error
          with COMPACT sources masked (galaxies are rare too and would
          otherwise dominate); tiles above the 99th percentile of the scores of
          20 % HELD-OUT tiles (never trained on) are flagged.

          Which tiles: only tiles in the Stage 0 train / dev sky (frame_qc.train_splits) fit, stop and
          calibrate the model; test, external and collar sky is only scored. The tiles are split into fit /
          validation / calibration by SKY CELL (frame_qc.split_cell_arcsec), not tile by tile: the three bands
          of a pointing (and overlapping files) see the same sky, so a per-tile shuffle put near-copies of a
          validation tile into training and made the validation loss and the threshold optimistic.

          qc_version 2 (default): a further 10 % of the tiles is a VALIDATION set. Training stops when the
          validation loss has not improved by min_delta for `patience` epochs (never before min_epochs, never
          while the model is < min_gain_vs_flat better than a flat tile, at most max_epochs) and keeps the
          epoch with the LOWEST validation loss (a fixed epoch count stopped the original model on its "flat
          tile" plateau: < 1 % better than a tile filled with its own mean), with the GroupNorm autoencoder of
          Stage 2a (no plateau), 90-degree rotations / flips, the reconstruction gain over a flat tile, and a
          bootstrap interval for the threshold and the flagged share. qc_version 1 reproduces the original
          (fixed steps).
Output  : tile_qc.csv (file, x0, y0, mse, flagged), qc_cae.pt, qc_threshold.json,
          qc_error_hist.png, qc_training.png / qc_training.json (v2) and one flag-map PNG per file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..context import Ctx, load_json, save_json
from ..fitsio import load_image, valid_mask
from ..nets import ConvAutoencoder, RepAutoencoder, train_autoencoder, train_rep_autoencoder
from ..utils import asinh_normalise
from .stage0_partition import SkyRegionPartitioner


def frame_sigma(sci, valid):
    d = sci[valid]
    if d.size > 2_000_000:
        d = d[:: d.size // 2_000_000]
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) or 1.0


def qc_normalise(tile, sigma, cap=200.0):
    """ONE fixed stretch for every tile of a file (in units of the sky noise).
    A per-tile percentile stretch would blow empty-sky noise up to full range and
    squash a satellite trail - exactly the opposite of what an anomaly detector needs."""
    x = np.arcsinh(np.clip(tile / (3.0 * sigma), 0, None)) / np.arcsinh(cap)
    return np.clip(x, 0, 1).astype(np.float32)


def source_masked_mse(x, recon, tile, level=0.2, min_extent_frac=0.5, max_fill=0.25):
    """Reconstruction error with SOURCE-LIKE structure masked.

    Galaxies are rare too, so a plain MSE mostly flags bright galaxies. Every
    connected structure above `level` is masked unless it looks like an artefact:
      * long and thin  (extent >= half a tile, filling < 25% of its box) -> trail
      * very large     (area >= a quarter of the tile)                   -> bad patch
    so the score measures what the CAE cannot explain OUTSIDE ordinary sources."""
    from scipy import ndimage
    errs = np.empty(len(x), np.float32)
    for i in range(len(x)):
        img = x[i, 0]
        lab, n = ndimage.label(img > level)
        keep = np.ones_like(img, bool)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            area = int((lab[sl] == k).sum())
            thin_long = max(h, w) >= min_extent_frac * tile and area / (h * w) < max_fill
            big = area >= 0.25 * tile * tile
            if not (thin_long or big):
                keep[sl] &= lab[sl] != k
        d = (img - recon[i, 0]) ** 2
        errs[i] = d[keep].mean() if keep.any() else 0.0
    return errs


@torch.no_grad()
def reconstruct(model, X, device, batch=256):
    model.to(device).eval()
    out = []
    for i in range(0, len(X), batch):
        r, _ = model(torch.tensor(X[i:i + batch]).to(device))
        out.append(r.cpu().numpy())
    return np.concatenate(out)


def tiles(sci, valid, size, stride):
    ny, nx = sci.shape
    for y0 in range(0, ny - size + 1, stride):
        for x0 in range(0, nx - size + 1, stride):
            cov = valid[y0:y0 + size, x0:x0 + size].mean()
            yield x0, y0, cov


def split_by_sky_cell(cells, cal_frac, val_frac, rng):
    """Whole sky cells (not single tiles) go to calibration / validation / fit, so the same patch of sky seen
    in another band, or by an overlapping file, never sits on both sides. Cells are taken in random order
    until each role holds its share of the tiles. Returns the role of every tile: 'cal', 'val' or 'fit'."""
    keys, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    order = rng.permutation(len(keys))
    start = (np.cumsum(counts[order]) - counts[order]) / counts.sum()    # tile share before each cell
    role = np.empty(len(keys), object)
    role[order] = np.where(start < cal_frac, "cal", np.where(start < cal_frac + val_frac, "val", "fit"))
    return role[inv.ravel()]


def run(ctx: Ctx):
    cfg = ctx.cfg["frame_qc"]
    out = ctx.stage_dir("1b")
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    rules = SkyRegionPartitioner.from_file(ctx.require("0", "split_rule.json"))
    files = sorted({f for p in pts.values() for f in p["bands"].values()})
    size, rng = cfg["tile"], np.random.default_rng(ctx.seed)
    device = ctx.device()
    fit_splits = set(cfg.get("train_splits", ["train", "dev"]))
    cell_deg = float(cfg.get("split_cell_arcsec", 30.0)) / 3600.0

    # ---- 1. well-covered tiles from the Stage 0 train / dev sky only (test, external and collar sky is
    #         scored in step 2 but never used to fit, stop or calibrate the QC model) ----
    train, cells, tags = [], [], []
    for f in files:
        sci, wht, w = load_image(f)
        val = valid_mask(sci, wht)
        sig = frame_sigma(sci, val)
        sci = np.where(val, sci - np.median(sci[val]), 0.0)
        pos = [(x0, y0) for x0, y0, cov in tiles(sci, val, size, cfg["train_stride"])
               if cov >= cfg["min_tile_coverage"]]
        if not pos:
            continue
        ra, dec = w.pixel_to_world_values(*(np.asarray(pos, float) + (size - 1) / 2).T)
        for (x0, y0), r, d, t in zip(pos, ra, dec, rules.assign_many(ra, dec)):
            tags.append(t or "collar")
            train.append(qc_normalise(sci[y0:y0 + size, x0:x0 + size], sig))
            cells.append((int(np.floor(r * np.cos(np.deg2rad(d)) / cell_deg)), int(np.floor(d / cell_deg))))
    tag_counts = pd.Series(tags).value_counts().to_dict()
    use = np.isin(tags, list(fit_splits))
    if not use.any():          # e.g. a field that is wholly external: nothing else to learn 'normal sky' from
        ctx.log.info(f"[Stage 1b] WARNING: no well-covered tile in the Stage 0 {sorted(fit_splits)} sky "
                     f"(tiles per split: {tag_counts}); the QC model uses every tile of the field")
        fit_splits, use = set(tag_counts), np.ones(len(tags), bool)
    train, cells = [x for x, u in zip(train, use) if u], [c for c, u in zip(cells, use) if u]
    if len(train) > cfg["max_train_tiles"]:
        pick = rng.choice(len(train), cfg["max_train_tiles"], replace=False)
        train, cells = [train[i] for i in pick], [cells[i] for i in pick]
    X = np.stack(train)[:, None]
    # 20 % of the tiles are HELD OUT: the threshold is set on tiles the CAE never trained on. A threshold
    # from its own training tiles is biased low (they are reconstructed best) and flags ordinary sky.
    version = int(cfg.get("qc_version", 2))
    role = split_by_sky_cell(np.asarray(cells), float(cfg.get("cal_fraction", 0.2)),
                             float(cfg.get("val_fraction", 0.1)) if version == 2 else 0.0, rng)
    n_cells = len(set(cells))
    if min((role == r).sum() for r in (("cal", "val", "fit") if version == 2 else ("cal", "fit"))) < 20:
        raise RuntimeError(f"[Stage 1b] {len(X)} tiles in {n_cells} sky cells cannot fill the fit / validation / "
                           "calibration sets (>= 20 tiles each); lower frame_qc.split_cell_arcsec")
    ctx.log.info(f"[Stage 1b] tiles per Stage 0 split: {tag_counts}; QC model uses {sorted(fit_splits)} only: "
                 f"{len(X)} tiles in {n_cells} sky cells of {cell_deg * 3600:g}\" (whole cells go to one role)")
    split_info = {"train_splits": sorted(fit_splits), "split_cell_arcsec": cell_deg * 3600, "n_sky_cells": n_cells,
                  "tiles_per_stage0_split": tag_counts}
    if version == 1:
        Xcal, Xfit = X[role == "cal"], X[role != "cal"]
        ctx.log.info(f"[Stage 1b] training QC-CAE v1 on {len(Xfit)} tiles from {len(files)} file(s) ({device}); "
                     f"{len(Xcal)} held-out tiles set the threshold")
        model = ConvAutoencoder(in_ch=1, latent_dim=cfg["latent_dim"], size=size)
        steps_per_epoch = int(np.ceil(len(Xfit) / cfg["batch_size"]))
        epochs = max(int(cfg.get("epochs", 8)), int(np.ceil(cfg.get("min_steps", 0) / steps_per_epoch)))
        ctx.log.info(f"[Stage 1b] {epochs} epochs x {steps_per_epoch} steps = {epochs * steps_per_epoch} optimisation steps")
        train_autoencoder(model, Xfit, epochs, cfg["batch_size"], device, log=ctx.log.info, tag="QC-CAE")
        extra = {"qc_version": 1, "epochs_run": epochs}
    else:
        # a further 10 % VALIDATION tiles decide when training stops; the 20 % calibration tiles still set the
        # threshold and are never used for training or stopping
        Xcal, Xval, Xfit = X[role == "cal"], X[role == "val"], X[role == "fit"]
        model, extra = _train_v2(ctx, cfg, Xfit, Xval, size, device, out)
        ctx.log.info(f"[Stage 1b] {len(Xfit)} training, {len(Xval)} validation and {len(Xcal)} calibration tiles "
                     f"from {len(files)} file(s)")
    train_err = source_masked_mse(Xfit, reconstruct(model, Xfit, device), size)
    cal_err = source_masked_mse(Xcal, reconstruct(model, Xcal, device), size)
    thr = float(np.percentile(cal_err, cfg["threshold_percentile"]))
    # the 99th percentile of a few hundred tiles rests on a handful of them: report its uncertainty
    boot = np.random.default_rng(ctx.seed).choice(cal_err, (1000, len(cal_err)), replace=True)
    thr_ci = np.percentile(np.percentile(boot, cfg["threshold_percentile"], axis=1), [2.5, 97.5])
    torch.save(model.state_dict(), out / "qc_cae.pt")
    qc_info = {"qc_version": version, "threshold_mse": thr, "threshold_ci95": [float(v) for v in thr_ci],
               "percentile": cfg["threshold_percentile"], "n_train_tiles": len(Xfit), "n_heldout_tiles": len(Xcal),
               "median_mse_train": float(np.median(train_err)), "median_mse_heldout": float(np.median(cal_err)),
               **split_info, **extra}
    ctx.log.info(f"[Stage 1b] threshold {thr:.5f} (95 % bootstrap interval {thr_ci[0]:.5f}-{thr_ci[1]:.5f})")

    # ---- 2. score every tile (overlapping) of every file ----
    rows = []
    for f in files:
        sci, wht, _ = load_image(f)
        val = valid_mask(sci, wht)
        sig = frame_sigma(sci, val)
        sci = np.where(val, sci - np.median(sci[val]), 0.0)
        pos, batch = [], []
        for x0, y0, cov in tiles(sci, val, size, cfg["score_stride"]):
            if cov < 0.05:            # empty border: nothing to judge
                continue
            pos.append((x0, y0, cov))
            batch.append(qc_normalise(sci[y0:y0 + size, x0:x0 + size], sig))
        if not batch:
            ctx.log.info(f"[Stage 1b] WARNING: {Path(f).name} has no tile with data - not scored")
            continue
        Xs = np.stack(batch)[:, None]
        err = source_masked_mse(Xs, reconstruct(model, Xs, device), size)
        for (x0, y0, cov), e in zip(pos, err):
            rows.append({"file": f, "x0": x0, "y0": y0, "size": size, "coverage": round(cov, 3),
                         "mse": float(e), "flagged": bool(e > thr)})
        _flag_map(f, sci, pos, err, thr, size, out)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tile_qc.csv", index=False)
    qc_info["flagged_fraction"] = float(df["flagged"].mean())
    qc_info["flagged_fraction_ci95"] = [float((df["mse"] > t).mean()) for t in thr_ci[::-1]]
    save_json(qc_info, out / "qc_threshold.json")
    _hist(cal_err, df["mse"].to_numpy(), thr, out / "qc_error_hist.png")
    frac = df.groupby("file")["flagged"].mean().round(4).to_dict()
    ctx.log.info(f"[Stage 1b] threshold MSE = {thr:.5f}; flagged tile fraction per file: "
                 f"{ {Path(k).name: v for k, v in frac.items()} }; overall {qc_info['flagged_fraction']:.2%} "
                 f"(threshold interval -> {qc_info['flagged_fraction_ci95'][0]:.2%}-{qc_info['flagged_fraction_ci95'][1]:.2%})")
    return {"threshold": thr, "flagged_fraction": float(df["flagged"].mean()), "qc_version": version,
            **({"epochs_run": extra.get("epochs_run"), "best_epoch": extra.get("best_epoch"),
                "gain_vs_flat_tile": extra.get("gain_vs_flat_tile")} if version == 2 else {})}


def _hist(cal_err, all_err, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(all_err, bins=60, alpha=0.6, label="all scored tiles")
    ax.hist(cal_err, bins=60, alpha=0.6, label="held-out tiles (set the threshold)")
    ax.axvline(thr, color="r", ls="--", label=f"threshold {thr:.4f}")
    ax.set_xlabel("reconstruction MSE")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Stage 1b: frame QC")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _flag_map(f, sci, pos, err, thr, size, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    step = max(1, max(sci.shape) // 1024)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(asinh_normalise(sci[::step, ::step]), origin="lower", cmap="gray")
    for (x0, y0, _), e in zip(pos, err):
        if e > thr:
            ax.add_patch(Rectangle((x0 / step, y0 / step), size / step, size / step, fill=False, ec="red", lw=0.8))
    ax.set_title(f"{Path(f).name}: flagged tiles (red)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / (Path(f).name.replace(".gz", "").replace(".fits", "") + "_qc.png"), dpi=100)
    plt.close(fig)


def _train_v2(ctx, cfg, Xfit, Xval, size, device, out):
    """QC autoencoder v2: GroupNorm encoder (as Stage 2a), plain MSE, early stopping on the validation tiles."""
    torch.manual_seed(ctx.seed)
    np.random.seed(ctx.seed)
    model = RepAutoencoder(bands=1, latent_dim=int(cfg["latent_dim"]), size=size,
                           channels=tuple(cfg.get("channels", [16, 32, 64, 64])), mask_channels=False)
    X = np.concatenate([Xfit, Xval]).astype(np.float16)
    P = np.ones(X.shape, bool)
    tr, va = np.arange(len(Xfit)), np.arange(len(Xfit), len(X))
    ctx.log.info(f"[Stage 1b] training QC-CAE v2 ({sum(p.numel() for p in model.parameters()):,} weights) on "
                 f"{len(tr)} tiles, validation {len(va)} ({device}); at most {cfg.get('max_epochs', 150)} epochs, "
                 f"early stopping after {cfg.get('patience', 15)} epochs without a "
                 f"{float(cfg.get('min_delta', 0.005)):.1%} improvement")
    v = np.asarray(Xval, np.float32)
    flat = float(((v - v.mean(axis=(2, 3), keepdims=True)) ** 2).mean())     # a tile filled with its own mean
    hist = train_rep_autoencoder(model, X, P, None, tr, va, device, max_epochs=int(cfg.get("max_epochs", 150)),
                                 min_epochs=int(cfg.get("min_epochs", 20)), patience=int(cfg.get("patience", 15)),
                                 batch_size=int(cfg["batch_size"]), lr=float(cfg.get("lr", 1e-3)),
                                 lr_patience=int(cfg.get("lr_patience", 5)), augment=bool(cfg.get("augment", True)),
                                 min_delta=float(cfg.get("min_delta", 0.005)), checkpoint=out / "qc_cae.pt",
                                 no_stop_above=flat * (1 - float(cfg.get("min_gain_vs_flat", 0.02))),
                                 restore_lowest=True, log=ctx.log.info, tag="QC-CAE")
    gain = 1 - hist["best_val"] / flat
    ctx.log.info(f"[Stage 1b] best validation loss {hist['best_val']:.5f} at epoch {hist['best_epoch']} of "
                 f"{len(hist['val'])}; a flat tile (each tile = its mean) scores {flat:.5f} -> gain {gain:.1%}")
    if gain < 0.02:
        ctx.log.info("[Stage 1b] WARNING: the QC autoencoder is hardly better than a flat tile - its flags then mark "
                     "high-variance tiles rather than what the model cannot explain.")
    save_json(hist, out / "qc_training.json")
    _training_plot(hist, flat, out / "qc_training.png")
    model.to(device)
    return model, {"epochs_run": len(hist["val"]), "best_epoch": hist["best_epoch"],
                   "best_val_loss": hist["best_val"], "flat_tile_val_loss": flat, "gain_vs_flat_tile": round(gain, 4)}


def _training_plot(hist, flat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["train"]) + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ep, hist["train"], label="train (augmented)", color="#2E5496")
    ax.plot(ep, hist["val"], label="validation tiles", color="#E69F00")
    ax.axhline(flat, color="0.4", ls=":", lw=1, label="flat tile (each tile = its mean)")
    ax.axvline(hist["best_epoch"], color="0.5", ls="--", lw=0.8, label=f"best epoch {hist['best_epoch']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Stage 1b: QC autoencoder training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
