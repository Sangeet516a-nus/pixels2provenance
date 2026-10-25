"""Multi-band checks shared by Stage 1c (and read by later stages through sources.csv).

Three problems of single-exposure HST products are handled here:

1. Band registration. Bands taken by different programmes can carry different
   astrometric solutions (e.g. an 'a priori' pointing-only WCS against one refined
   on a star catalogue). `registration_catalog` detects sources on a MEDIAN-FILTERED
   image: a single exposure holds tens of thousands of cosmic rays, which are
   brighter than most galaxies and would fill any 'brightest sources' list, but
   they are only 1-3 px wide and a 5x5 median erases them while galaxies survive.
   `measure_offset` finds the shift from the pairs of the two lists and reports how
   far the peak stands above the random-pair level; `shift_wcs` removes it.
2. Cosmic rays (source level). Cosmic rays hit each exposure at random places,
   so a real source is also seen in a second band of the same sky, a cosmic ray
   almost never. `segment_photometry` measures every detection segment in every
   band (S/N from the background rms) and the centroid of the band's significant
   pixels on a 3x3 MEDIAN-FILTERED copy (cosmic rays erased, compact sources kept).
   A band CONFIRMS a source when its S/N >= detection.confirm_snr AND its light is
   centred where the detection-band light is (within detection.confirm_max_shift_px):
   in a single exposure a cosmic ray of the other band often lands inside a large
   footprint (high S/N at the wrong place), and cosmic rays swallowed by a galaxy's
   detection footprint would pull a raw centroid away.
3. Cosmic rays (pixel level). Inside a cutout, a pixel that is far brighter in
   one band than the other bands predict (given the source's own colour) is a
   cosmic-ray hit in that band. `cr_pixel_mask` finds them; Stage 1c replaces
   them by the prediction and marks them 3 in the segmentation stamp, so
   statmorph masks them.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- 1. registration
def tangent_arcsec(ra, dec, ra0, dec0):
    """Small-field tangent-plane offsets (arcsec) from (ra0, dec0): x = dRA cos(dec), y = dDec."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    return (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0, (dec - dec0) * 3600.0


def registration_catalog(sub, rms, valid, wcs, cfg, median_px=5, nsigma=5.0, npixels=25, n_max=3000, max_ellip=0.8):
    """RA/Dec of the brightest extended sources of a median-filtered image (cosmic rays removed)."""
    from .compat import SourceCatalog, n_labels, values
    from .detection import detect_and_deblend
    med = ndimage.median_filter(np.where(valid, sub, 0.0).astype(np.float32), size=median_px)
    sky = med[valid]
    if sky.size > 2_000_000:
        sky = sky[:: sky.size // 2_000_000]
    mad = 1.4826 * np.median(np.abs(sky - np.median(sky)))           # noise of the median image
    rms_m = rms * (mad / max(float(np.nanmedian(rms)), 1e-12))
    segm, conv = detect_and_deblend(med, rms_m, valid, {**cfg, "nsigma": nsigma, "npixels": npixels},
                                    deblend_sources=False)
    if n_labels(segm) == 0:
        return np.zeros(0), np.zeros(0)
    cat = SourceCatalog(med, segm, convolved_data=conv)
    x, y = values(cat, "x_centroid"), values(cat, "y_centroid")
    flux, ell = values(cat, "segment_flux"), values(cat, "ellipticity")
    ok = np.where(np.isfinite(x) & np.isfinite(flux) & (ell < max_ellip))[0]
    ok = ok[np.argsort(-flux[ok])][:n_max]
    return wcs.pixel_to_world_values(x[ok], y[ok])


def registration_sources(x, y, snr, semimajor, ellipticity, min_snr=20.0, min_sigma_px=1.2, max_ellip=0.7, n_max=3000):
    """Indices of sources good for registration: bright and EXTENDED (sharp cosmic rays and long
    thin tracks would match random partners in a crowded single exposure)."""
    ok = (np.isfinite(x) & np.isfinite(y) & (np.asarray(snr) >= min_snr) & (np.asarray(semimajor) >= min_sigma_px)
          & (np.asarray(ellipticity) < max_ellip))
    idx = np.where(ok)[0]
    return idx[np.argsort(-np.asarray(snr)[idx])][:n_max]


def measure_offset(ref_xy, band_xy, search_arcsec=3.0, bin_arcsec=0.05, refine_arcsec=0.15):
    """Median offset (band - reference, arcsec) from all pairs within `search_arcsec`.
    The peak of the 2-D histogram of pair offsets is robust to the random pairs of a crowded
    field; the median of the pairs within `refine_arcsec` of the peak refines it.
    Returns (dx, dy, n_pairs, scatter_arcsec, significance), significance = n_pairs / the number of
    random pairs expected in the same refine circle (a real offset gives >> 5)."""
    if len(ref_xy) == 0 or len(band_xy) == 0:
        return np.nan, np.nan, 0, np.nan, 0.0
    tree = cKDTree(band_xy)
    pairs = tree.query_ball_point(ref_xy, search_arcsec)
    d = np.array([band_xy[j] - ref_xy[i] for i, js in enumerate(pairs) for j in js], float).reshape(-1, 2)
    if len(d) < 5:
        return np.nan, np.nan, 0, np.nan, 0.0
    edges = np.arange(-search_arcsec, search_arcsec + bin_arcsec, bin_arcsec)
    H, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=[edges, edges])
    H = ndimage.uniform_filter(H, 3)                     # 3x3 smoothing: the peak is a few bins wide
    i, j = np.unravel_index(np.argmax(H), H.shape)
    peak = np.array([0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])])
    for _ in range(3):                                   # iterate the refinement around the median
        near = np.hypot(*(d - peak).T) <= refine_arcsec
        if near.sum() < 3:
            break
        peak = np.median(d[near], axis=0)
    near = np.hypot(*(d - peak).T) <= refine_arcsec
    scatter = float(1.4826 * np.median(np.hypot(*(d[near] - peak).T))) if near.any() else np.nan
    expected = len(d) * refine_arcsec ** 2 / search_arcsec ** 2          # random pairs in the refine circle
    return float(peak[0]), float(peak[1]), int(near.sum()), scatter, float(near.sum() / max(expected, 1e-9))


def shift_wcs(wcs, dx_arcsec, dy_arcsec):
    """Copy of `wcs` whose sky positions are moved by (-dx, -dy) arcsec: the band's positions were
    (dx, dy) too large (dx = dRA cos(dec)). Shifting CRVAL moves a small field rigidly."""
    w = wcs.deepcopy()
    dec0 = w.wcs.crval[1]
    w.wcs.crval = [w.wcs.crval[0] - dx_arcsec / 3600.0 / np.cos(np.deg2rad(dec0)), dec0 - dy_arcsec / 3600.0]
    w.wcs.set()
    return w


# ---------------------------------------------------------------- 2. source-level confirmation
def median_image(sub, valid, size=3):
    """Median-filtered copy and the factor by which its noise is lower than the input's (from the MAD)."""
    img = np.where(valid, sub, 0.0).astype(np.float32)
    med = ndimage.median_filter(img, size=size)
    a, m = img[valid], med[valid]
    step = max(1, a.size // 2_000_000)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    return med, float(mad(m[::step]) / max(mad(a[::step]), 1e-12))


def segment_photometry(labels, det_wcs, band, det_pixarea, max_label, corr=1.0, sig_level=2.0):
    """Flux, error and coverage of every detection segment measured in another band.

    labels   : detection-band segmentation image (0 = sky)
    band     : dict with 'sub', 'rms', 'val', 'wcs', 'pixarea' of the band
    Each detection pixel samples the band at the same sky position (nearest pixel); the flux is
    scaled by the pixel-area ratio. The centroid (detection-band pixel units) is weighted by the band's
    MEDIAN-FILTERED pixels (band['med'], noise factor band['med_noise']) above sig_level sigma only, so
    cosmic rays inside the footprint cannot move it. Returns arrays indexed by label (0..max_label):
    flux, err, coverage, cx, cy (NaN where the band has no significant pixel)."""
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    n_all = np.bincount(lab, minlength=max_label + 1).astype(float)
    if band is None:
        z = np.zeros(max_label + 1)
        nan = np.full(max_label + 1, np.nan)
        return z, nan, z, nan, nan
    ra, dec = det_wcs.pixel_to_world_values(xs, ys)
    bx, by = band["wcs"].world_to_pixel_values(ra, dec)
    del ra, dec
    xi, yi = np.round(bx).astype(np.int64), np.round(by).astype(np.int64)
    h, w = band["sub"].shape
    ok = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    ok[ok] = band["val"][yi[ok], xi[ok]]
    scale = det_pixarea / band["pixarea"]
    v = np.zeros(len(lab))
    var = np.zeros(len(lab))
    v[ok] = band["sub"][yi[ok], xi[ok]] * scale
    var[ok] = (band["rms"][yi[ok], xi[ok]] * scale * corr) ** 2
    flux = np.bincount(lab, weights=v, minlength=max_label + 1)
    err = np.sqrt(np.bincount(lab, weights=var, minlength=max_label + 1))
    cov = np.bincount(lab, weights=ok.astype(float), minlength=max_label + 1) / np.maximum(n_all, 1)
    vm = np.zeros(len(lab))
    vm[ok] = band["med"][yi[ok], xi[ok]] * scale
    wgt = np.where(vm > sig_level * np.sqrt(var) * band["med_noise"], vm, 0.0)
    wsum = np.bincount(lab, weights=wgt, minlength=max_label + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(wsum > 0, np.bincount(lab, weights=wgt * xs, minlength=max_label + 1) / wsum, np.nan)
        cy = np.where(wsum > 0, np.bincount(lab, weights=wgt * ys, minlength=max_label + 1) / wsum, np.nan)
    return flux, np.where(err > 0, err, np.nan), cov, cx, cy


# ---------------------------------------------------------------- 3. pixel-level cosmic-ray mask
_RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)


def cr_pixel_mask(cube, valid, seg, sigma, flux, snr, nsigma=5.0, ratio=4.0, min_ref_snr=3.0, sharp_ratio=2.0):
    """Cosmic-ray pixels of one cutout, per band, and the replacement values.

    cube  : (bands, S, S) raw stamps (e-/s per output pixel)
    valid : (bands, S, S) bool coverage
    seg   : (S, S) 0 sky / 1 target / 2 other source
    sigma : (bands, S, S) per-pixel noise (background + source Poisson)
    flux, snr : per-band segment flux and S/N of the TARGET (fallback colour for faint targets)
    A pixel of band b is a hit when it is > nsigma above the value that EVERY usable other band c
    predicts (v_c x R_bc) and more than `ratio` times that prediction. R_bc, the target's colour, is the
    MEDIAN pixel ratio v_b / v_c over the target pixels bright in c (> 5 sigma): segment fluxes are no good
    here, because cosmic-ray hits inside a segment inflate them and the whole galaxy would then look too
    bright in the other band. Band c is usable when the target is detected in it (snr >= min_ref_snr) and
    it covers the pixel. Pixels of other sources (seg == 2) are not tested: their colour is not the
    target's, and they are masked anyway.
    A hit must also be SHARP in its own band: more than nsigma above the median of its 8 neighbours and
    more than sharp_ratio times that median. Without this, the smooth red halo of ACS F814W around bright
    stars (scattered light the blue bands do not have) is 'too bright for its colour' and was masked.
    Returns (mask (bands, S, S) bool, replacement (bands, S, S))."""
    nb = cube.shape[0]
    mask = np.zeros(cube.shape, bool)
    repl = np.zeros(cube.shape, np.float32)
    test_area = seg != 2
    target = seg == 1
    for b in range(nb):
        preds, errs = [], []
        for c in range(nb):
            if c == b or not (np.isfinite(snr[c]) and snr[c] >= min_ref_snr and flux[c] > 0 and np.isfinite(flux[b])):
                continue
            bright = target & valid[b] & valid[c] & (cube[c] > 5 * sigma[c])
            if bright.sum() >= 5:
                r = float(np.median(cube[b][bright] / cube[c][bright]))
            else:
                r = max(flux[b], 0.0) / flux[c]
            r = max(r, 0.0)
            preds.append(np.where(valid[c], cube[c] * r, np.nan))
            errs.append(np.where(valid[c], sigma[c] * r, np.nan))
        if not preds:
            continue
        P, E = np.stack(preds), np.stack(errs)
        excess = cube[b][None] - P
        tot = np.sqrt(sigma[b][None] ** 2 + E ** 2)
        hit_c = (excess > nsigma * tot) & (cube[b][None] > ratio * np.clip(P, 0, None))
        usable = np.isfinite(P)
        hit = np.all(hit_c | ~usable, axis=0) & usable.any(axis=0)       # every usable band disagrees
        hit &= valid[b] & test_area & (cube[b] > nsigma * sigma[b])
        if hit.any():
            ring = ndimage.median_filter(cube[b], footprint=_RING, mode="nearest")
            hit &= (cube[b] - ring > nsigma * sigma[b]) & (cube[b] > sharp_ratio * np.clip(ring, 0, None))
        if hit.any():
            hit = ndimage.binary_dilation(hit, iterations=1) & valid[b] & test_area   # CR wings
            mask[b] = hit
            repl[b] = np.nanmedian(np.where(usable, P, np.nan), axis=0)
    return mask, np.nan_to_num(repl)


# ---------------------------------------------------------------- exposures per pixel
def popcount(ctx_values):
    """Number of set bits of the drizzle context (CTX) = number of input images at that pixel.
    For ACS/WFC the two chips of ONE exposure never overlap, so this counts exposures."""
    v = np.asarray(ctx_values).astype(np.uint64)
    n = np.zeros(v.shape, np.int64)
    while np.any(v):
        n += (v & np.uint64(1)).astype(np.int64)
        v = v >> np.uint64(1)
    return n
