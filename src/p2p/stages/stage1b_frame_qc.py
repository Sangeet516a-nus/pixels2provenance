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
Output  : tile_qc.csv (file, x0, y0, mse, flagged), qc_cae.pt, qc_threshold.json,
          qc_error_hist.png and one flag-map PNG per file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..context import Ctx, load_json, save_json
from ..fitsio import load_image, valid_mask
from ..nets import ConvAutoencoder, train_autoencoder
from ..utils import asinh_normalise


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


def tiles(sci, valid, size, stride, min_cov):
    ny, nx = sci.shape
    for y0 in range(0, ny - size + 1, stride):
        for x0 in range(0, nx - size + 1, stride):
            cov = valid[y0:y0 + size, x0:x0 + size].mean()
            yield x0, y0, cov


def run(ctx: Ctx):
    cfg = ctx.cfg["frame_qc"]
    out = ctx.stage_dir("1b")
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    files = sorted({f for p in pts.values() for f in p["bands"].values()})
    size, rng = cfg["tile"], np.random.default_rng(ctx.seed)
    device = ctx.device()

    # ---- 1. training sample of well-covered tiles from every file ----
    train = []
    for f in files:
        sci, wht, _ = load_image(f)
        val = valid_mask(sci, wht)
        sig = frame_sigma(sci, val)
        sci = np.where(val, sci - np.median(sci[val]), 0.0)
        for x0, y0, cov in tiles(sci, val, size, cfg["train_stride"], cfg["min_tile_coverage"]):
            if cov >= cfg["min_tile_coverage"]:
                train.append(qc_normalise(sci[y0:y0 + size, x0:x0 + size], sig))
    if len(train) > cfg["max_train_tiles"]:
        train = [train[i] for i in rng.choice(len(train), cfg["max_train_tiles"], replace=False)]
    X = np.stack(train)[:, None]
    # 20 % of the tiles are HELD OUT: the threshold is set on tiles the CAE never trained on. A threshold
    # from its own training tiles is biased low (they are reconstructed best) and flags ordinary sky.
    perm = rng.permutation(len(X))
    n_cal = max(int(round(0.2 * len(X))), min(20, len(X) // 2))
    Xcal, Xfit = X[perm[:n_cal]], X[perm[n_cal:]]
    ctx.log.info(f"[Stage 1b] training QC-CAE on {len(Xfit)} tiles from {len(files)} file(s) ({device}); "
                 f"{len(Xcal)} held-out tiles set the threshold")
    model = ConvAutoencoder(in_ch=1, latent_dim=cfg["latent_dim"], size=size)
    steps_per_epoch = int(np.ceil(len(Xfit) / cfg["batch_size"]))
    epochs = max(int(cfg["epochs"]), int(np.ceil(cfg.get("min_steps", 0) / steps_per_epoch)))
    ctx.log.info(f"[Stage 1b] {epochs} epochs x {steps_per_epoch} steps = {epochs * steps_per_epoch} optimisation steps")
    train_autoencoder(model, Xfit, epochs, cfg["batch_size"], device, log=ctx.log.info, tag="QC-CAE")
    train_err = source_masked_mse(Xfit, reconstruct(model, Xfit, device), size)
    cal_err = source_masked_mse(Xcal, reconstruct(model, Xcal, device), size)
    thr = float(np.percentile(cal_err, cfg["threshold_percentile"]))
    torch.save(model.state_dict(), out / "qc_cae.pt")
    save_json({"threshold_mse": thr, "percentile": cfg["threshold_percentile"], "n_train_tiles": len(Xfit),
               "n_heldout_tiles": len(Xcal), "median_mse_train": float(np.median(train_err)),
               "median_mse_heldout": float(np.median(cal_err))},
              out / "qc_threshold.json")

    # ---- 2. score every tile (overlapping) of every file ----
    rows = []
    for f in files:
        sci, wht, _ = load_image(f)
        val = valid_mask(sci, wht)
        sig = frame_sigma(sci, val)
        sci = np.where(val, sci - np.median(sci[val]), 0.0)
        pos, batch = [], []
        for x0, y0, cov in tiles(sci, val, size, cfg["score_stride"], 0):
            if cov < 0.05:            # empty border: nothing to judge
                continue
            pos.append((x0, y0, cov))
            batch.append(qc_normalise(sci[y0:y0 + size, x0:x0 + size], sig))
        Xs = np.stack(batch)[:, None]
        err = source_masked_mse(Xs, reconstruct(model, Xs, device), size)
        for (x0, y0, cov), e in zip(pos, err):
            rows.append({"file": f, "x0": x0, "y0": y0, "size": size, "coverage": round(cov, 3),
                         "mse": float(e), "flagged": bool(e > thr)})
        _flag_map(f, sci, val, pos, err, thr, out)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tile_qc.csv", index=False)
    _hist(cal_err, df["mse"].to_numpy(), thr, out / "qc_error_hist.png")
    frac = df.groupby("file")["flagged"].mean().round(4).to_dict()
    ctx.log.info(f"[Stage 1b] threshold MSE = {thr:.5f}; flagged tile fraction per file: "
                 f"{ {Path(k).name: v for k, v in frac.items()} }")
    return {"threshold": thr, "flagged_fraction": float(df["flagged"].mean())}


def _hist(train_err, all_err, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(all_err, bins=60, alpha=0.6, label="all scored tiles")
    ax.hist(train_err, bins=60, alpha=0.6, label="held-out tiles (set the threshold)")
    ax.axvline(thr, color="r", ls="--", label=f"threshold {thr:.4f}")
    ax.set_xlabel("reconstruction MSE")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Stage 1b: frame QC")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _flag_map(f, sci, val, pos, err, thr, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    step = max(1, max(sci.shape) // 1024)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(asinh_normalise(sci[::step, ::step]), origin="lower", cmap="gray")
    for (x0, y0, _), e in zip(pos, err):
        if e > thr:
            ax.add_patch(Rectangle((x0 / step, y0 / step), 128 / step, 128 / step, fill=False, ec="red", lw=0.8))
    ax.set_title(f"{Path(f).name}: flagged tiles (red)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / (Path(f).name.replace(".gz", "").replace(".fits", "") + "_qc.png"), dpi=100)
    plt.close(fig)
