"""Stage 4a - CNN feature extraction: ResNet-18 + EfficientNet-B0 (ImageNet weights, frozen).

Purpose : describe every galaxy's image by the visual traits two pretrained CNNs see (edges, curves, textures,
          gradients). No CNN is trained here - with ~1,100 labelled galaxies a trained CNN memorises them; the
          only model that learns from the labels is XGBoost (Stage 4b).
Input   : every Galaxy-track cutout (Stage 2e; Stage 1d's PSF-matched 128 x 128 x 3 cutouts):
          cosmic rays removed (the Stage 9 cleaner) -> pixels of neighbouring sources (Stage 1c segmentation)
          replaced by sky -> crop of cnn_crop_petro (4) x the Petrosian radius (>= 32 px) so the galaxy fills the
          image -> one asinh stretch (same for all bands, colours kept) -> R, G, B = F775W, F606W, F435W
          -> ImageNet mean / std -> bilinear resize to supervised.input_size (224); features averaged over
          cnn_tta_views (4) rotations by 90 deg (a galaxy's class does not depend on its orientation).
Models  : ResNet-18       conv 7x7 -> 4 x 2 residual blocks (64-128-256-512) -> GlobalAveragePooling -> 512
          EfficientNet-B0 conv 3x3 -> 16 MBConv blocks -> conv 1x1 (1280)  -> GlobalAveragePooling -> 1,280
          Activations ReLU (ResNet) / SiLU (EfficientNet), BatchNorm with ImageNet statistics, Dropout off;
          the ImageNet classifiers (1,000 classes) are not used.
Output  : cnn_features.npy (n_galaxies x 1,792: [ResNet-18 512 | EfficientNet-B0 1,280]), object_id.npy,
          feature_info.json, model_summary.txt
"""
from __future__ import annotations

import time

import numpy as np

from ..context import Ctx, save_json
from ..features import cnn_stamp, galaxy_table, imagenet_input, load_cutouts
from ..inference import CNN_BLOCKS, build_extractors, extract_cnn_features
from ..nets import model_summary


def run(ctx: Ctx):
    from ..modes import check_bands, cnn_input_settings, is_apply
    cfg = cnn_input_settings(ctx)
    out = ctx.stage_dir("4a")
    if is_apply(ctx):                       # the frozen CNNs need no bundle - only the same band set and the same
        check_bands(ctx)                    # CNN input settings (crop, rotations, ...), taken from the bundle
        ctx.log.info("[Stage 4a] apply mode: CNN input settings from the model bundle: "
                     + ", ".join(f"{k}={cfg.get(k)}" for k in ("input_size", "cnn_crop", "cnn_crop_petro", "cnn_crop_min",
                                                             "cnn_mask_neighbours", "cnn_clean_cosmic_rays",
                                                             "cnn_tta_views")))
    size, pretrained = int(cfg.get("input_size", 224)), bool(cfg.get("pretrained", True))
    gal = galaxy_table(ctx)
    ctx.log.info(f"[Stage 4a] {len(gal):,} galaxies -> ImageNet input {size} x {size} -> frozen ResNet-18 + "
                 f"EfficientNet-B0 ({'ImageNet weights' if pretrained else 'RANDOM weights (pretrained: false)'}) "
                 f"on {ctx.device()}")

    # ---- 1. build the two frozen networks  2. summary ----
    nets = build_extractors(pretrained)
    r0 = gal.iloc[0]
    seg0 = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")[int(r0.cutout_index)]
    x = imagenet_input([cnn_stamp(load_cutouts(ctx)[int(r0.cutout_index)], seg0, r0.get("r_petro"), r0.get("area_px"),
                                  cfg, ctx.cfg.get("atlas", {}))], size)
    summ = "\n\n".join(model_summary(nets[k], x, name=f"Stage 4a - {k} (ImageNet, frozen)", max_depth=None)
                       for k, _ in CNN_BLOCKS)
    (out / "model_summary.txt").write_text(summ + "\n")
    ctx.log.info("[Stage 4a] model summaries:\n" + summ)

    # ---- 3. extract: one forward pass per cutout, no training ----
    t0 = time.time()
    ctx.log.info(f"[Stage 4a] CNN input: crop {cfg.get('cnn_crop', True)} ({cfg.get('cnn_crop_petro', 4.0)} x r_petro, "
                 f">= {cfg.get('cnn_crop_min', 32)} px), neighbour mask {cfg.get('cnn_mask_neighbours', True)}, "
                 f"cosmic-ray cleaning {cfg.get('cnn_clean_cosmic_rays', True)}, {cfg.get('cnn_tta_views', 4)} rotations")
    F = extract_cnn_features(ctx, gal, nets, log=ctx.log.info)
    np.save(out / "cnn_features.npy", F)
    np.save(out / "object_id.npy", gal.object_id.to_numpy())
    blocks, s = {}, 0
    for k, n in CNN_BLOCKS:
        blocks[k] = [s, s + n]
        s += n
    info = {"n_galaxies": int(len(gal)), "n_features": int(F.shape[1]), "blocks": blocks, "input_size": size,
            "pretrained": pretrained, "channels": "R, G, B = reddest, middle, bluest band (asinh, ImageNet mean/std)",
            "cnn_input": {k: cfg.get(k) for k in ("cnn_crop", "cnn_crop_petro", "cnn_crop_min", "cnn_mask_neighbours",
                                                  "cnn_clean_cosmic_rays", "cnn_tta_views")},
            "feature_mean_by_block": {k: round(float(F[:, a:b].mean()), 4) for k, (a, b) in blocks.items()},
            "seconds": round(time.time() - t0, 1)}
    save_json(info, out / "feature_info.json")
    ctx.log.info(f"[Stage 4a] features {F.shape} ({info['seconds']} s): ResNet-18 columns 0-511, EfficientNet-B0 "
                 f"columns 512-1791 -> {ctx.rel(out / 'cnn_features.npy')}")
    return {"n_galaxies": int(len(gal)), "n_features": int(F.shape[1]), "input_size": size}
