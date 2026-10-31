"""Stage 9 - Galaxy image atlas: a colour PNG of every galaxy in the catalogue.

Purpose : let a person look at every classified galaxy, and hand the images on
          together with catalog.csv.
Method  : for each catalogue row with classification == Galaxy, the registered,
          cosmic-ray-cleaned cutout of Stage 1c (native resolution, all bands)
          is cleaned of the cosmic rays left outside the target's footprint (display
          only - see clean_cosmic_rays), background-subtracted (median of the cutout border), put on one
          flux scale with the AB zero-points of its pointing, and combined into an
          RGB image (reddest band = R, bluest = B; with two bands G is their mean,
          with one band the image is grey) with a colour-preserving asinh stretch
          (Lupton et al. 2004): the softening is set from the sky noise of the
          border and the top from the bright end of the galaxy. Images are saved
          as PNG with a 256-colour palette (compressed), in one folder per
          predicted class, and written to galaxy_images.zip together with
          galaxy_images_index.csv, which links every image to its catalogue row
          (object_id, designation, class, probabilities, GZH label).
          gallery_<class>.png shows the most confident galaxies of each class;
          galaxy_atlas_overview.png shows a few of every class. A catalogue galaxy
          whose central light is mostly removed by the cleaner is a cosmic ray that
          reached the galaxy track: it is flagged cr_suspect in the index, left out
          of the galleries and shown (uncleaned) in gallery_cr_suspects.png.
Output  : images/<class>/<designation>.png, galaxy_images_index.csv,
          galaxy_images.zip, gallery_<class>.png, galaxy_atlas_overview.png,
          gallery_cr_suspects.png, atlas_summary.json
"""
from __future__ import annotations

import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from ..utils import galaxy_track
from ..context import Ctx, StageSkipped, load_json, save_json

CLASSES = ["Smooth", "FaceOnSpiral", "EdgeOn", "Irregular", "Ambiguous"]
PCOLS = {"Smooth": "p_smooth", "FaceOnSpiral": "p_spiral", "EdgeOn": "p_edgeon",
         "Irregular": "p_irregular", "Ambiguous": "p_ambiguous"}


def _wavelength(band: str) -> float:
    """F475W -> 475, F110W (WFC3/IR) -> 1100: HST filter names give the pivot in nm (IR in 10 nm)."""
    m = re.search(r"(\d+)", band)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v * 10 if v < 200 else v          # F105W..F160W are IR filters: 1050..1600 nm


def _border(a: np.ndarray, w: int = 8) -> np.ndarray:
    """Border pixels of a stamp that have data (uncovered pixels are NaN or exactly 0 in the cutouts)."""
    b = np.concatenate([a[:w].ravel(), a[-w:].ravel(), a[w:-w, :w].ravel(), a[w:-w, -w:].ravel()])
    b = b[np.isfinite(b) & (b != 0)]
    return b if b.size >= 20 else np.array([0.0, 1.0])


def _covered(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & (a != 0)


def clean_cosmic_rays(cut: np.ndarray, order: list[int], cfg: dict) -> np.ndarray:
    """Display-only cosmic-ray removal (the measurements of Stages 1c-8 are not touched).
    Single exposures leave many cosmic rays OUTSIDE the target footprint (Stage 1c cleans the target's
    footprint only). A pixel of band b is replaced when it is (1) sharp in b: > nsigma sky-sigma above the
    median of its 8 neighbours and > 2x that median, and (2) a single-band event: its S/N in b is more than
    `cr_ratio` times its S/N in every other band. Real sources are smooth at the PSF scale (FWHM ~2.6 px),
    so they fail (1), and they appear in every band, so they fail (2). The mask is grown by one pixel and
    replaced with the 5x5 median of the band; two passes catch the inside of wide tracks."""
    from scipy import ndimage
    ring = np.ones((3, 3), bool); ring[1, 1] = False
    nsig, ratio = float(cfg.get("cr_nsigma", 5.0)), float(cfg.get("cr_ratio", 3.0))
    cov = np.stack([_covered(np.asarray(cut[b], np.float32)) for b in range(cut.shape[0])])
    a = np.nan_to_num(np.asarray(cut, np.float32), nan=0.0).copy()
    for _ in range(int(cfg.get("cr_passes", 2))):
        sig = np.array([1.4826 * np.median(np.abs(_border(a[b]) - np.median(_border(a[b])))) or 1e-6 for b in range(a.shape[0])])
        snr = np.stack([(a[b] - np.median(_border(a[b]))) / sig[b] for b in range(a.shape[0])])
        for b in order:
            ringmed = ndimage.median_filter(a[b], footprint=ring, mode="nearest")
            sharp = (a[b] - ringmed > nsig * sig[b]) & (a[b] > 2 * np.maximum(ringmed, 0) )
            oth = [c for c in order if c != b]
            if oth:
                # only bands that COVER the pixel can say 'nothing is there'; where no other band covers it,
                # the pixel cannot be tested and is left alone (chip gaps, frame edges)
                o_snr = np.stack([np.where(cov[c], np.clip(snr[c], 1.0, None), np.nan) for c in oth])
                any_cov = np.isfinite(o_snr).any(axis=0)
                others = np.nanmax(np.where(np.isfinite(o_snr), o_snr, -np.inf), axis=0)
                single = any_cov & (snr[b] > ratio * others)
            else:
                single = np.ones_like(sharp)
            sharp &= cov[b]
            seed = sharp & single
            # hysteresis: grow each sharp seed through connected single-band pixels (S/N > 3), so the inside of
            # long, wide tracks (not sharp against their own neighbours) goes too; a red galaxy has no seed
            lab, n = ndimage.label(single & (snr[b] > 3.0))
            if n:
                hit_lab = np.unique(lab[seed & (lab > 0)])
                seed = seed | np.isin(lab, hit_lab[hit_lab > 0])
            m = ndimage.binary_dilation(seed, iterations=1)
            if m.any():
                med5 = ndimage.median_filter(np.where(m, np.nan, a[b]), size=5, mode="nearest")
                fill = np.where(np.isfinite(med5), med5, np.median(_border(a[b])))
                a[b] = np.where(m, fill, a[b])
    return a


def removed_fraction(before: np.ndarray, after: np.ndarray, half: int = 6) -> float:
    """Share of the significant light (> 2 sky sigma) in the central (2*half)^2 pixels, summed over bands,
    that the cleaner removed."""
    h, w = before.shape[-2:]
    tot = rem = 0.0
    for i in range(before.shape[0]):
        b0 = np.nan_to_num(np.asarray(before[i], np.float32))
        a0 = np.nan_to_num(np.asarray(after[i], np.float32))
        sky = _border(b0)
        bg = np.median(sky)
        sig = 1.4826 * np.median(np.abs(sky - bg)) or 1e-6
        c0 = b0[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half] - bg
        c1 = a0[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half] - bg
        sig_px = c0 > 2 * sig
        tot += c0[sig_px].sum()
        rem += np.clip(c0 - c1, 0, None)[sig_px].sum()
    return float(np.clip(rem / tot, 0, 1)) if tot > 0 else 0.0


def colour_image(cut: np.ndarray, order: list[int], scale: np.ndarray, cfg: dict) -> np.ndarray:
    """cut: (bands, H, W) in e-/s; order: channel indices from bluest to reddest; scale: AB flux factor per band.
    Returns uint8 RGB (H, W, 3)."""
    chans = []
    for b in order:
        raw_b = np.asarray(cut[b], np.float32)
        a = np.nan_to_num(raw_b, nan=0.0)
        a = np.where(_covered(raw_b), (a - np.median(_border(a))) * scale[b], 0.0)
        chans.append(a)
    if len(chans) >= 3:
        bl, gr, rd = chans[0], chans[len(chans) // 2], chans[-1]
    elif len(chans) == 2:
        bl, rd = chans
        gr = 0.5 * (bl + rd)
    else:
        bl = gr = rd = chans[0]
    inten = (bl + gr + rd) / 3.0
    sky = _border(inten)
    sigma = 1.4826 * np.median(np.abs(sky - np.median(sky))) or float(np.std(sky)) or 1e-6
    h, w = inten.shape
    core = inten[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    top = max(float(np.percentile(core, cfg.get("top_percentile", 99.7))), cfg.get("min_top_sigma", 15) * sigma)
    soft = cfg.get("soft_sigma", 2.0) * sigma
    f = np.arcsinh(np.clip(inten, 0, None) / soft) / np.arcsinh(top / soft)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(inten > 0, f / inten, 0.0)
    rgb = np.stack([rd * k, gr * k, bl * k], axis=-1)
    mx = rgb.max(axis=-1, keepdims=True)                    # keep the hue when a channel saturates
    rgb = np.where(mx > 1, rgb / np.maximum(mx, 1e-9), rgb)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_png(rgb: np.ndarray, path, cfg: dict):
    im = Image.fromarray(rgb, "RGB")
    up = int(cfg.get("upscale", 1))
    if up > 1:
        im = im.resize((im.width * up, im.height * up), Image.NEAREST)
    colours = int(cfg.get("palette_colours", 256))
    if 2 <= colours <= 256:
        im = im.quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(path, format="PNG", optimize=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9+\-.]+", "_", str(s)).strip("_")


def gallery(rows: pd.DataFrame, images: dict, title: str, path, ncol: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    nrow = max(1, int(np.ceil(n / ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 2.15 * nrow + 0.5), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axs.flat, rows.iterrows()):
        img = images[r.object_id]
        ax.imshow(np.asarray(Image.open(img).convert("RGB")) if not isinstance(img, np.ndarray) else img,
                  interpolation="nearest")
        ax.set_title(f"{str(r.designation).replace('P2P ', '')}\n{r.pred_class} p={r.p_max:.2f}", fontsize=6.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def run(ctx: Ctx):
    cfg = ctx.cfg.get("atlas", {})
    if not cfg.get("enabled", True):
        raise StageSkipped("atlas.enabled is false in config.yaml")
    out = ctx.stage_dir("9")
    cat = pd.read_csv(ctx.require("8", "catalog.csv"), low_memory=False)
    gal = cat[galaxy_track(cat)].copy()           # Galaxy rows without deblending fragments (fragment_of)
    if cfg.get("min_snr"):
        gal = gal[gal.snr >= float(cfg["min_snr"])]
    if gal.empty:
        raise StageSkipped("no galaxies in catalog.csv")
    pc = [PCOLS[c] for c in CLASSES if PCOLS[c] in gal]
    gal["p_max"] = gal[pc].max(axis=1)
    gal["pred_class"] = gal.pred_class.fillna("Unclassified")
    if cfg.get("max_galaxies"):
        gal = gal.sort_values("p_max", ascending=False).head(int(cfg["max_galaxies"]))

    src = pd.read_csv(ctx.require("1c", "sources.csv"), usecols=["object_id", "cutout_index", "pointing"])
    gal = gal.merge(src.rename(columns={"pointing": "_pt"}), on="object_id", how="left")
    raw = np.load(ctx.require("1c", "cutouts_raw.npy"), mmap_mode="r")
    info = load_json(ctx.require("1a", "pointings.json"))
    bands = info["bands"]
    order = sorted(range(len(bands)), key=lambda i: _wavelength(bands[i]))

    img_dir = out / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    images, paths, cr_frac = {}, [], []
    for _, r in gal.iterrows():
        pt = info["pointings"].get(r._pt, {})
        zp = pt.get("zp_ab", {})
        ref = zp.get(pt.get("detection_band"), 25.0) if zp else 25.0
        scale = np.array([10 ** (-0.4 * (zp.get(b, ref) - ref)) if b in zp else 1.0 for b in bands], np.float32)
        have = [i for i in order if b_ok(bands[i], pt)]
        stamp = np.asarray(raw[int(r.cutout_index)], np.float32)
        removed = 0.0
        if cfg.get("clean_cosmic_rays", True):
            clean = clean_cosmic_rays(stamp, have or order, cfg)
            removed = removed_fraction(stamp, clean)
            stamp = clean
        cr_frac.append(removed)
        rgb = colour_image(stamp, have or order, scale, cfg)
        # 31 Oct: a prediction below atlas.uncertain_below goes to Uncertain/ (the class is still in the file name
        # and the index); a galaxy in a Stage 1b QC-flagged tile gets _QC in its file name
        unsure = r.p_max < float(cfg.get("uncertain_below", 0.4))
        cls_dir = img_dir / ("Uncertain" if unsure else safe_name(r.pred_class))
        cls_dir.mkdir(parents=True, exist_ok=True)
        tag = (f"_{safe_name(r.pred_class)}_p{r.p_max:.2f}" if unsure else "") + \
              ("_QC" if str(r.get("frame_quality_flag", 1)) in ("0", "0.0") else "")
        p = cls_dir / f"{safe_name(r.designation)}{tag}.png"
        save_png(rgb, p, cfg)
        images[r.object_id] = p                 # the PNG path: 15,000 RGB arrays would need ~0.7 GB of memory
        paths.append(str(p.relative_to(out)))
    gal["image"] = paths
    gal["cr_removed_frac"] = np.round(cr_frac, 3)
    # most of the target's own central light was a single-band, sharp event: a cosmic ray, not a galaxy
    gal["cr_suspect"] = gal.cr_removed_frac > float(cfg.get("cr_suspect_fraction", 0.5))

    keep = ["object_id", "designation", "ra", "dec", "pointing", "pred_class", "p_max"] + pc + \
           [c for c in ("pred_entropy", "p_merger", "gz_label", "is_pseudo_label", "mag", "snr", "split_tag") if c in gal] + \
           ["cr_removed_frac", "cr_suspect", "image"]
    idx = gal[keep].sort_values(["pred_class", "p_max"], ascending=[True, False])
    idx.to_csv(out / "galaxy_images_index.csv", index=False)

    zpath = out / "galaxy_images.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(out / "galaxy_images_index.csv", "galaxy_images_index.csv")
        for p in paths:
            z.write(out / p, p)
    n_gal = int(cfg.get("gallery_n", 36))
    ncol = int(cfg.get("gallery_columns", 6))
    per_class = {}
    good = idx[~idx.cr_suspect]
    for c in CLASSES + ["Unclassified"]:
        sub = good[good.pred_class == c].head(n_gal)
        per_class[c] = int((idx.pred_class == c).sum())
        if len(sub):
            gallery(sub, images, f"Stage 9: {c} — the {len(sub)} most confident of {per_class[c]} galaxies",
                    out / f"gallery_{c}.png", ncol)
    per_row = int(cfg.get("overview_per_class", 6))
    ov = pd.concat([good[good.pred_class == c].head(per_row) for c in CLASSES if (good.pred_class == c).any()])
    if len(ov):
        gallery(ov, images, "Stage 9: galaxy atlas overview (most confident per class)",
                out / "galaxy_atlas_overview.png", per_row)
    sus = idx[idx.cr_suspect].sort_values("cr_removed_frac", ascending=False).head(n_gal)
    if len(sus):
        raw_imgs = {}
        for _, r in sus.iterrows():
            k = int(gal.loc[gal.object_id == r.object_id, "cutout_index"].iloc[0])
            pt = info["pointings"].get(gal.loc[gal.object_id == r.object_id, "_pt"].iloc[0], {})
            zp = pt.get("zp_ab", {}); ref = zp.get(pt.get("detection_band"), 25.0) if zp else 25.0
            scale = np.array([10 ** (-0.4 * (zp.get(b, ref) - ref)) if b in zp else 1.0 for b in bands], np.float32)
            raw_imgs[r.object_id] = colour_image(np.asarray(raw[k], np.float32), order, scale, cfg)
        gallery(sus, raw_imgs, f"Stage 9: catalogue 'galaxies' that look like cosmic rays "
                f"({int(idx.cr_suspect.sum())}; uncleaned view, check before use)", out / "gallery_cr_suspects.png", ncol)

    size_mb = sum((out / p).stat().st_size for p in paths) / 1e6
    if not cfg.get("keep_png_folder", True):             # galleries are drawn: the PNGs live on in the zip
        shutil.rmtree(img_dir)
    summary = {"galaxies_imaged": len(paths), "per_class": per_class, "bands_rgb_order_blue_to_red": [bands[i] for i in order],
               "png_total_mb": round(size_mb, 2),
               "zip_mb": round(zpath.stat().st_size / 1e6, 2), "palette_colours": int(cfg.get("palette_colours", 256)),
               "cr_suspects": int(gal.cr_suspect.sum()),
               "cr_suspect_rule": f"> {cfg.get('cr_suspect_fraction', 0.5)} of the central light removed as single-band, sharp pixels"}
    save_json(summary, out / "atlas_summary.json")
    ctx.log.info(f"[Stage 9] {len(paths)} galaxy images ({summary['png_total_mb']} MB PNG, zip {summary['zip_mb']} MB) "
                 f"| per class {per_class} | RGB = {summary['bands_rgb_order_blue_to_red'][::-1]} "
                 f"| look like cosmic rays: {summary['cr_suspects']} (galaxy_images_index.csv: cr_suspect)")
    return {"galaxies_imaged": len(paths), "zip_mb": summary["zip_mb"]}


def b_ok(band: str, pointing: dict) -> bool:
    """A band the pointing actually has (missing bands are left out of the colour image)."""
    return band not in pointing.get("missing_bands", [])
