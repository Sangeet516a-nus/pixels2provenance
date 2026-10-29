"""Stage 1c - Detection, band registration, cosmic-ray tests and aligned cutouts (no learned model).

Per pointing:
  1. Background-subtract every band; detect and deblend sources on the detection band (photutils).
  2. Register each band to the detection band: offsets measured on median-filtered images (cosmic rays erased)
     are removed from the band's WCS; the stage stops if the offset cannot be measured or a residual remains.
  3. Measure every source in every band. A source that no other covering band confirms (S/N >= confirm_snr,
     light at the same position) is a cosmic-ray candidate. Confirmed sources and candidates are capped separately.
  4. Cut a north-up 128 x 128 stamp per source from every band at the same RA/Dec (affine resampling).
     Cosmic-ray pixels (far brighter in one band than the source colour predicts) are replaced and marked.
  5. Flags: edge (WHT coverage), frame QC (Stage 1b tiles), saturation, exposures (CTX); split from Stage 0.
Output  : sources.csv, cutouts_raw.npy (N, bands, 128, 128, float16), segstamps.npy (N, 128, 128, uint8:
          1 target, 2 neighbour, 3 cosmic-ray pixel), registration.json, summary.json, preview PNGs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from photutils.utils import calc_total_error
from scipy import ndimage

from ..compat import SourceCatalog, n_labels, values
from ..context import Ctx, load_json, save_json
from ..detection import detect_and_deblend, subtract_background
from ..fitsio import load_ctx, load_image, read_meta, valid_mask
from ..multiband import (cr_pixel_mask, measure_offset, median_image, popcount, registration_catalog,
                         segment_photometry, shift_wcs, tangent_arcsec)
from .stage0_partition import SkyRegionPartitioner


def affine_for(wcs, ra0, dec0, size, scale_arcsec):
    """ndimage.affine_transform matrix/offset from a north-up cutout (row, col) to the band's pixels, and the
    pixel-area ratio."""
    c = (size - 1) / 2.0
    cosd = np.cos(np.deg2rad(dec0))
    # cutout pixel (u, v) -> sky: east is left (RA grows to -u), north is up
    def sky(du, dv):
        return ra0 - du * scale_arcsec / 3600.0 / cosd, dec0 + dv * scale_arcsec / 3600.0
    pts = [sky(0, 0), sky(1, 0), sky(0, 1)]
    xy = np.array([wcs.world_to_pixel_values(r, d) for r, d in pts], float)
    x0, y0 = xy[0]
    J = np.array([[xy[1, 0] - x0, xy[2, 0] - x0],          # dx/du dx/dv
                  [xy[1, 1] - y0, xy[2, 1] - y0]])         # dy/du dy/dv
    M = np.array([[J[1, 1], J[1, 0]],                      # (row, col) order
                  [J[0, 1], J[0, 0]]])
    offset = np.array([y0, x0]) - M @ np.array([c, c])
    return M, offset, abs(np.linalg.det(J))


def make_stamp(img, M, offset, size, order=1):
    return ndimage.affine_transform(img, M, offset=offset, output_shape=(size, size),
                                    order=order, mode="constant", cval=0.0, prefilter=False)


def flagged_tiles(ctx: Ctx):
    """Stage 1b flagged tiles as their central stride x stride cell (tiles overlap, so each pixel is judged by
    exactly one tile per band)."""
    p = ctx.path("1b", "tile_qc.csv")
    if not p.exists():
        return {}
    t = pd.read_csv(p)
    t = t[t["flagged"].astype(str).str.lower() == "true"]
    stride = int(ctx.cfg["frame_qc"].get("score_stride", 64))
    m = (t["size"] - stride) / 2.0
    t = t.assign(cx0=t.x0 + m, cy0=t.y0 + m, cs=np.minimum(stride, t["size"]))
    return {f: g[["cx0", "cy0", "cs"]].to_numpy(float) for f, g in t.groupby("file")}


def in_flagged(x, y, boxes):
    if boxes is None or len(boxes) == 0:
        return np.zeros(len(x), bool)
    x, y = np.asarray(x)[:, None], np.asarray(y)[:, None]
    x0, y0, s = boxes[:, 0][None], boxes[:, 1][None], boxes[:, 2][None]
    return np.any((x >= x0) & (x < x0 + s) & (y >= y0) & (y < y0 + s), axis=1)


def saturated_cores(sub, rms, x, y, half=4, tol=0.02, min_peak_sigma=50.0):
    """Flat-topped cores on native pixels: the 5 brightest pixels within +-4 px agree to 2 %, peak >= 50 sigma."""
    h, w = sub.shape
    out = np.zeros(len(x), bool)
    for k, (xc, yc) in enumerate(zip(np.round(x).astype(int), np.round(y).astype(int))):
        y0, y1, x0, x1 = max(0, yc - half), min(h, yc + half + 1), max(0, xc - half), min(w, xc + half + 1)
        box = sub[y0:y1, x0:x1]
        if box.size < 5:
            continue
        top = np.sort(box.ravel())[-5:]
        out[k] = top[-1] > min_peak_sigma * rms[min(h - 1, max(0, yc)), min(w - 1, max(0, xc))] and \
            (top[-1] - top[0]) / top[-1] < tol
    return out


def register_bands(ctx, pid, imgs, det_b, dcfg):
    """Measure and remove each band's sky offset against the detection band (sources from median-filtered
    images, so cosmic rays do not pair at random). Returns the per-band report."""
    det = imgs[det_b]
    ra0, dec0 = det["wcs"].wcs.crval
    limit = float(dcfg.get("max_band_offset_arcsec", 0.05))
    search = float(dcfg.get("register_search_arcsec", 3.0))
    min_sig = float(dcfg.get("register_min_significance", 5.0))
    manual = {str(k).upper(): v for k, v in (dcfg.get("band_offsets_arcsec") or {}).items()}
    rr, rd = registration_catalog(det["sub"], det["rms"], det["val"], det["wcs"], dcfg)
    ref_xy = np.column_stack(tangent_arcsec(rr, rd, ra0, dec0))
    report = {}
    for b, im in imgs.items():
        if b == det_b:
            continue
        br, bd = registration_catalog(im["sub"], im["rms"], im["val"], im["wcs"], dcfg)
        bpx = im["wcs"].world_to_pixel_values(br, bd)        # pixel positions: fixed, only the WCS changes

        def offset(wcs):
            r, d = wcs.pixel_to_world_values(*bpx)
            return measure_offset(ref_xy, np.column_stack(tangent_arcsec(r, d, ra0, dec0)), search)

        dx, dy, n, scat, sig = offset(im["wcs"])
        reliable = n >= 20 and sig >= min_sig
        rep = {"n_reference": len(ref_xy), "n_band": len(br), "n_pairs": n, "significance": round(sig, 1),
               "offset_measured_arcsec": [round(dx, 4), round(dy, 4)] if n else None,
               "scatter_arcsec": round(scat, 4) if n else None, "reliable": bool(reliable), "applied": None}
        if b in manual:                                  # offset given in config.yaml
            ox, oy = (float(v) for v in manual[b])
            rep["applied"] = {"source": "config detection.band_offsets_arcsec", "offset_arcsec": [ox, oy]}
        elif reliable and dcfg.get("register_bands", True) and np.hypot(dx, dy) > 0.01:
            ox, oy = dx, dy
            rep["applied"] = {"source": "measured", "offset_arcsec": [round(ox, 4), round(oy, 4)]}
        if rep["applied"]:
            im["wcs"] = shift_wcs(im["wcs"], ox, oy)
            dx, dy, n2, _, sig2 = offset(im["wcs"])
            reliable = n2 >= 20 and sig2 >= min_sig
            rep.update(n_pairs_after=n2, significance_after=round(sig2, 1))
        rep["offset_after_arcsec"] = [round(dx, 4), round(dy, 4)] if reliable else None
        rep["residual_arcsec"] = round(float(np.hypot(dx, dy)), 4) if reliable else None
        report[b] = rep
        ctx.log.info(f"[Stage 1c] {pid}: {b} vs {det_b}: {rep['n_pairs']} matched sources (peak {rep['significance']}x the "
                     f"random-pair level), offset {rep['offset_measured_arcsec']}\" -> "
                     f"{'corrected (' + rep['applied']['source'] + ')' if rep['applied'] else 'no correction'}, "
                     f"residual {rep['offset_after_arcsec']}\" (dRA cos(dec), dDec)")
        if not reliable:
            raise RuntimeError(
                f"[Stage 1c] {pid}: the offset of {b} against {det_b} cannot be measured reliably ({rep['n_pairs']} "
                f"pairs, {rep['significance']}x the random level; need >= 20 pairs and >= {min_sig}x). Colours and "
                f"two-band tests would use unchecked positions. If you know the offset, give it in config.yaml, e.g. "
                f"detection.band_offsets_arcsec: {{{b}: [dRA_cos_dec, dDec]}} (arcsec, band minus {det_b}).")
        if np.hypot(dx, dy) > limit:
            raise RuntimeError(
                f"[Stage 1c] {pid}: {b} is still offset by {np.hypot(dx, dy):.3f}\" from {det_b} after registration "
                f"(limit detection.max_band_offset_arcsec = {limit}\"). "
                + ("Turn on detection.register_bands." if not dcfg.get("register_bands", True) else
                   "Check detection.band_offsets_arcsec, or use a file whose WCS is aligned to a star catalogue."))
    return report


def process_pointing(ctx, pid, p, bands, rules, qc_boxes, pidx):
    dcfg = ctx.cfg["detection"]
    S = dcfg["cutout_size"]
    corr = dcfg.get("correlated_noise_factor", 1.0)
    imgs = {}
    for b, f in p["bands"].items():
        sci, wht, w = load_image(f)
        val = valid_mask(sci, wht)
        sub, rms = subtract_background(sci, val, dcfg["background_box"], dcfg["background_filter"])
        imgs[b] = {"sub": sub, "rms": rms, "val": val, "valf": val.astype(np.float32), "wcs": w, "file": f,
                   "scale": p["pixscale"][b], "pixarea": p["pixscale"][b] ** 2, "zp": p["zp_ab"][b],
                   "exptime": max(float(read_meta(Path(f))["exptime"]), 1.0)}
        del sci, wht
    det_b = p["detection_band"]
    det = imgs[det_b]
    segm, conv = detect_and_deblend(det["sub"], det["rms"], det["val"], dcfg)
    if n_labels(segm) == 0:
        ctx.log.info(f"[Stage 1c] {pid}: no sources detected")
        return None, None, None, None
    err = calc_total_error(det["sub"], det["rms"] * corr, effective_gain=max(p["exptime_detection"], 1.0))
    cat = SourceCatalog(det["sub"], segm, convolved_data=conv, error=err)
    x, y = values(cat, "x_centroid"), values(cat, "y_centroid")
    kron = values(cat, "kron_flux")
    kerr = values(cat, "kron_flux_err")
    segf = values(cat, "segment_flux")
    flux = np.where(np.isfinite(kron) & (kron > 0), kron, segf)
    ferr = np.where(np.isfinite(kerr) & (kerr > 0), kerr, values(cat, "segment_flux_err"))
    ra, dec = det["wcs"].pixel_to_world_values(x, y)
    tab = pd.DataFrame({
        "label": np.asarray(cat.labels, int), "x": x, "y": y, "ra": ra, "dec": dec,
        "area_px": values(cat, "area"), "semimajor_px": values(cat, "semimajor_axis"),
        "semiminor_px": values(cat, "semiminor_axis"), "ellipticity": values(cat, "ellipticity"),
        "orientation_deg": values(cat, "orientation"), "fwhm_px": values(cat, "fwhm"),
        "flux": flux, "flux_err": ferr, "snr": flux / np.where(ferr > 0, ferr, np.nan)})
    tab = tab[np.isfinite(tab.x) & np.isfinite(tab.ra) & (tab.flux > 0)]
    tab["saturated"] = saturated_cores(det["sub"], det["rms"], tab.x.to_numpy(), tab.y.to_numpy())

    # ---- 1. register every band to the detection band (fails on a residual offset) ----
    registration = register_bands(ctx, pid, imgs, det_b, dcfg)

    # ---- 2. every segment measured in every band -> cosmic-ray candidates (no second band confirms) ----
    lab_img = segm.data
    max_lab = int(lab_img.max())
    li = tab.label.to_numpy()
    confirm = float(dcfg.get("confirm_snr", 3.0))
    # allowed centroid shift between bands: 1.5 px, or half the equivalent radius for large galaxies
    max_shift = np.maximum(float(dcfg.get("confirm_max_shift_px", 1.5)),
                           float(dcfg.get("confirm_max_shift_frac", 0.5)) * np.sqrt(tab.area_px.to_numpy() / np.pi))
    best_other = np.full(len(tab), np.nan)
    n_cover_other = np.zeros(len(tab), int)
    for b, im in imgs.items():                       # cosmic-ray-free copies for the centroid test
        im["med"], im["med_noise"] = median_image(im["sub"], im["val"], 3)
    phot = {b: segment_photometry(lab_img, det["wcs"], imgs.get(b), det["pixarea"], max_lab, corr) for b in bands}
    dcx, dcy = phot[det_b][3][li], phot[det_b][4][li]
    for b in bands:
        fl, er, cov, cx, cy = phot[b]
        tab[f"segflux_{b}"] = fl[li]
        tab[f"segsnr_{b}"] = (fl / er)[li]
        tab[f"segcov_{b}"] = cov[li]
        tab[f"segshift_{b}"] = np.hypot(cx[li] - dcx, cy[li] - dcy)       # centroid shift vs detection band [px]
        if b != det_b and b in imgs:
            covered = cov[li] >= 0.9
            n_cover_other += covered
            # a band confirms only with light at the same position
            snr_here = np.where(tab[f"segshift_{b}"].to_numpy() <= max_shift, (fl / er)[li], np.nan)
            best_other = np.fmax(best_other, np.where(covered, snr_here, np.nan))
    tab["confirm_snr_best"] = best_other
    # True = no other covering band sees it; False = confirmed; None = no other band covers it
    use_cr = str(dcfg.get("cr_rejection", "two_band")).lower() != "none"
    tab["cr_candidate"] = [None if (not use_cr or n == 0) else bool(not (np.isfinite(s) and s >= confirm))
                           for n, s in zip(n_cover_other, best_other)]
    is_cr = tab.cr_candidate.map(lambda v: v is True).astype(bool)
    ctx.log.info(f"[Stage 1c] {pid}: {len(tab)} detections | confirmed by another band {int((~is_cr & (n_cover_other > 0)).sum())} "
                 f"| cosmic-ray candidates {int(is_cr.sum())} | no other band covers {int((n_cover_other == 0).sum())}")

    # ---- 3. caps: real (confirmed / unknown) sources and cosmic-ray candidates separately ----
    cap, cap_cr = int(dcfg["max_sources_per_pointing"]), int(dcfg.get("max_cr_candidates_per_pointing", 1000))
    real, crs = tab[~is_cr], tab[is_cr]
    if len(real) > cap:
        ctx.log.info(f"[Stage 1c] {pid}: {len(real)} confirmed sources, keeping the brightest {cap} "
                     "(detection.max_sources_per_pointing)")
        real = real.nlargest(cap, "flux")
    if len(crs) > cap_cr:
        ctx.log.info(f"[Stage 1c] {pid}: {len(crs)} cosmic-ray candidates, keeping the brightest {cap_cr} "
                     "(detection.max_cr_candidates_per_pointing; they stay in the catalogue as Noise)")
        crs = crs.nlargest(cap_cr, "flux")
    tab = pd.concat([real, crs]).sort_values("label").reset_index(drop=True)

    # ---- 4. aligned stamps; cosmic-ray pixels replaced and marked 3 ----
    seg_f = lab_img.astype(np.float32)          # converted once, sampled per source
    scale = det["scale"] if dcfg["cutout_pixscale_arcsec"] == "auto" else float(dcfg["cutout_pixscale_arcsec"])
    cubes = np.zeros((len(tab), len(bands), S, S), np.float16)
    segs = np.zeros((len(tab), S, S), np.uint8)
    yy, xx = np.mgrid[:S, :S]
    core = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2) <= dcfg["core_radius_px"]
    grow = int(dcfg.get("cr_mask_grow_px", 1))
    mag_snr = float(dcfg.get("mag_min_snr", 2.0))
    rows = []
    for i, r in enumerate(tab.itertuples()):
        n_nb = 0
        raw = np.zeros((len(bands), S, S), np.float32)
        sig = np.full((len(bands), S, S), np.inf, np.float32)
        val = np.zeros((len(bands), S, S), bool)
        for bi, b in enumerate(bands):
            if b not in imgs:
                continue
            im = imgs[b]
            M, off, area_ratio = affine_for(im["wcs"], r.ra, r.dec, S, scale)
            raw[bi] = make_stamp(im["sub"], M, off, S) * area_ratio
            val[bi] = make_stamp(im["valf"], M, off, S, order=0) > 0.5
            bx, by = im["wcs"].world_to_pixel_values(r.ra, r.dec)
            yi = int(np.clip(round(float(by)), 0, im["rms"].shape[0] - 1))
            xi = int(np.clip(round(float(bx)), 0, im["rms"].shape[1] - 1))
            rms_px = float(im["rms"][yi, xi]) * area_ratio * corr
            sig[bi] = np.sqrt(rms_px ** 2 + np.clip(raw[bi], 0, None) * area_ratio / im["exptime"])
            if b == det_b:
                lab = make_stamp(seg_f, M, off, S, order=0).astype(np.int64)
                segs[i][(lab > 0) & (lab != r.label)] = 2
                segs[i][lab == r.label] = 1
                n_nb = len(set(np.unique(lab).tolist()) - {0, r.label})
        valid_det = val[bands.index(det_b)]
        target = segs[i] == 1                     # full target footprint, before the cosmic-ray mask
        sflux = np.array([getattr(r, f"segflux_{b}") for b in bands], float)
        ssnr = np.array([getattr(r, f"segsnr_{b}") for b in bands], float)
        n_cr_target = 0
        if use_cr and len(imgs) > 1:
            crm, repl = cr_pixel_mask(raw, val, segs[i], sig, sflux, ssnr, float(dcfg.get("cr_pixel_nsigma", 5.0)),
                                      float(dcfg.get("cr_pixel_ratio", 4.0)))
            raw = np.where(crm, repl, raw)
            anyc = crm.any(axis=0)
            if anyc.any():
                if grow > 0:                      # PSF matching (Stage 1d) spreads a residual a little
                    anyc = ndimage.binary_dilation(anyc, iterations=grow)
                anyc &= segs[i] != 2
                n_cr_target = int((anyc & target).sum())
                segs[i][anyc] = 3
        cubes[i] = raw.astype(np.float16)
        row = {"valid_frac": float(valid_det.mean()), "core_ok": bool(valid_det[core].all()), "n_neighbours": n_nb,
               "cr_pixels_masked": n_cr_target}
        for bi, b in enumerate(bands):
            covered = b in imgs and bool(val[bi][core].all())          # this band has data on the source core
            row[f"covered_{b}"] = covered
            # flux over the whole target footprint, cosmic-ray pixels replaced
            fl = float(raw[bi][target & val[bi]].astype(np.float64).sum()) if covered else np.nan
            row[f"flux_{b}"] = fl
            ok = covered and fl > 0 and np.isfinite(imgs[b]["zp"]) and np.nan_to_num(ssnr[bi]) >= mag_snr
            row[f"mag_{b}"] = -2.5 * np.log10(fl) + imgs[b]["zp"] if ok else np.nan
            e = sflux[bi] / ssnr[bi] if np.isfinite(ssnr[bi]) else np.nan
            row[f"magerr_{b}"] = 1.0857 * e / fl if ok and np.isfinite(e) else np.nan
        rows.append(row)
    tab = pd.concat([tab, pd.DataFrame(rows)], axis=1)
    tab["object_id"] = (pidx + 1) * 100000 + tab["label"]
    tab["pointing"] = pid
    tab["role"] = p["role"]
    tab["detection_band"] = det_b
    tab["pixscale_out"] = scale
    tab["exptime"] = p["exptime_detection"]
    tab["mag"] = -2.5 * np.log10(tab["flux"]) + (det["zp"] if np.isfinite(det["zp"]) else 0.0)
    # statistical centroid error = source size / SNR per axis (absolute WCS error not included)
    pos_err = tab.fwhm_px / 2.3548 / tab.snr * det["scale"]
    tab["ra_err"] = pos_err.round(5)
    tab["dec_err"] = pos_err.round(5)
    ctx_img = load_ctx(det["file"])
    if ctx_img is not None:
        xi = np.clip(np.round(tab.x).astype(int), 0, ctx_img.shape[1] - 1)
        yi = np.clip(np.round(tab.y).astype(int), 0, ctx_img.shape[0] - 1)
        tab["n_exposures"] = popcount(ctx_img[yi, xi])
    else:
        tab["n_exposures"] = np.nan
    tab["edge_flag"] = (tab["valid_frac"] < dcfg["min_valid_fraction"]) | (~tab["core_ok"])
    bad = np.zeros(len(tab), bool)
    bad_bands = [[] for _ in range(len(tab))]
    for b, im in imgs.items():                 # a source is QC-flagged if ANY band's tile is flagged
        bx, by = im["wcs"].world_to_pixel_values(tab.ra.to_numpy(), tab.dec.to_numpy())
        hit = in_flagged(bx, by, qc_boxes.get(im["file"]))
        bad |= hit
        for j in np.where(hit)[0]:
            bad_bands[j].append(b)
    tab["frame_quality_flag"] = (~bad).astype(int)          # 1 = clean, 0 = in a flagged tile
    tab["qc_flagged_bands"] = [",".join(v) for v in bad_bands]
    tab["bands_missing"] = ",".join(p["missing_bands"])
    split = rules.assign_many(tab.ra, tab.dec)
    if p["role"] == "external":
        split = ["external" if s == "external" else None for s in split]
    tab["split_tag"] = [s if s is not None else "buffer" for s in split]     # buffer = no split (collar)
    hit = tab.cr_pixels_masked[tab.cr_pixels_masked > 0]
    ctx.log.info(f"[Stage 1c] {pid}: cosmic-ray pixels masked in {len(hit)} of {len(tab)} target footprints "
                 f"(median {hit.median() if len(hit) else 0} px)")
    _preview(ctx, pid, det, tab, cubes, bands)
    return tab, cubes, segs, registration


def run(ctx: Ctx):
    out = ctx.stage_dir("1c")
    info = load_json(ctx.require("1a", "pointings.json"))
    bands, pts = info["bands"], info["pointings"]
    rules = SkyRegionPartitioner.from_file(ctx.require("0", "split_rule.json"))
    qc = flagged_tiles(ctx)
    parts, registration = [], {}
    for pidx, (pid, p) in enumerate(pts.items()):
        ctx.log.info(f"[Stage 1c] {pid}: detecting on {p['detection_band']} ...")
        tab, cubes, segs, reg = process_pointing(ctx, pid, p, bands, rules, qc, pidx)
        if tab is None:
            continue
        registration[pid] = reg
        tmp = out / f"_tmp_{pid}"
        np.save(f"{tmp}_cut.npy", cubes)
        np.save(f"{tmp}_seg.npy", segs)
        parts.append((tab, tmp))
        ctx.log.info(f"[Stage 1c] {pid}: {len(tab)} sources | median SNR {tab.snr.median():.1f} | "
                     f"edge-flagged {int(tab.edge_flag.sum())} | QC-flagged {int((tab.frame_quality_flag == 0).sum())} | "
                     f"splits {tab.split_tag.value_counts(dropna=False).to_dict()}")
    if not parts:
        raise RuntimeError("No sources detected in any pointing.")

    # ---- merge pointings into single arrays ----
    n = sum(len(t) for t, _ in parts)
    S = ctx.cfg["detection"]["cutout_size"]
    cut = np.lib.format.open_memmap(out / "cutouts_raw.npy", mode="w+", dtype=np.float16, shape=(n, len(bands), S, S))
    seg = np.lib.format.open_memmap(out / "segstamps.npy", mode="w+", dtype=np.uint8, shape=(n, S, S))
    k, tabs = 0, []
    for tab, tmp in parts:
        c, s = np.load(f"{tmp}_cut.npy"), np.load(f"{tmp}_seg.npy")
        cut[k:k + len(tab)], seg[k:k + len(tab)] = c, s
        tab["cutout_index"] = np.arange(k, k + len(tab))
        tabs.append(tab)
        k += len(tab)
        Path(f"{tmp}_cut.npy").unlink()
        Path(f"{tmp}_seg.npy").unlink()
    cut.flush()
    seg.flush()
    sources = pd.concat(tabs, ignore_index=True)
    bl = list(bands)
    sources["colour_1"] = sources[f"mag_{bl[0]}"] - sources[f"mag_{bl[1]}"] if len(bl) > 1 else np.nan
    sources["colour_2"] = sources[f"mag_{bl[1]}"] - sources[f"mag_{bl[2]}"] if len(bl) > 2 else np.nan
    # band consistency: weakest other-band flux / detection-band flux, over the bands that cover the source
    # (a real source shows in every filter, a cosmic ray in one)
    det_flux = sources.apply(lambda r: r[f"flux_{r.detection_band}"], axis=1)
    ratios = [(sources[f"flux_{b}"] / det_flux).where((sources.detection_band != b) & sources[f"covered_{b}"])
              for b in bl]
    sources["band_consistency"] = np.clip(pd.concat(ratios, axis=1).min(axis=1), 0, 2) if len(bl) > 1 else np.nan
    front = ["object_id", "pointing", "role", "split_tag", "ra", "dec", "x", "y", "mag", "snr", "flux", "flux_err"]
    sources = sources[front + [c for c in sources.columns if c not in front]]
    sources.to_csv(out / "sources.csv", index=False)
    save_json(registration, out / "registration.json")
    save_json({"bands": bands, "cutout_size": S, "n_sources": n,
               "split_counts": sources["split_tag"].value_counts(dropna=False).to_dict()}, out / "summary.json")
    ctx.log.info(f"[Stage 1c] {n} sources total -> sources.csv, cutouts_raw.npy {cut.shape}")
    return {"n_sources": n}


def _preview(ctx, pid, det, tab, cubes, bands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..utils import asinh_normalise
    out = ctx.stage_dir("1c")
    step = max(1, max(det["sub"].shape) // 1024)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(asinh_normalise(det["sub"][::step, ::step]), origin="lower", cmap="gray")
    sc = ax.scatter(tab.x / step, tab.y / step, s=8, c=np.log10(tab.snr.clip(1)), cmap="viridis",
                    edgecolors="none")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="log10 SNR")
    ax.set_title(f"Stage 1c {pid}: {len(tab)} detections on {tab.detection_band.iloc[0]}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / f"{pid}_detections.png", dpi=100)
    plt.close(fig)
    # 16 brightest stamps as colour composites (red = reddest band)
    pick = tab.sort_values("flux", ascending=False).index[:16]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, i in zip(axes.ravel(), pick):
        cube = cubes[i].astype(np.float32)
        rgb = np.stack([asinh_normalise(cube[min(j, len(bands) - 1)]) for j in (2, 1, 0)], -1)
        ax.imshow(rgb, origin="lower")
        ax.set_title(f"id {tab.loc[i, 'label']}  SNR {tab.loc[i, 'snr']:.0f}", fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{pid}: brightest cutouts (aligned, {len(bands)} bands)")
    fig.tight_layout()
    fig.savefig(out / f"{pid}_cutouts.png", dpi=100)
    plt.close(fig)
