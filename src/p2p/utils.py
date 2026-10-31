"""Numerical helpers shared by several stages."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- images
def asinh_normalise(img, q_lo=1.0, q_hi=99.5, soft=0.1):
    """Percentile-clipped asinh stretch to [0, 1]; robust to hot pixels."""
    p = np.nan_to_num(np.asarray(img, dtype=np.float32))
    lo, hi = np.percentile(p, [q_lo, q_hi])
    x = np.clip((p - lo) / max(hi - lo, 1e-8), 0, 1)
    return (np.arcsinh(x / soft) / np.arcsinh(1.0 / soft)).astype(np.float32)


def asinh_stack(cubes):
    """(N, C, H, W) -> asinh-normalised float32, each band of each object separately."""
    cubes = np.asarray(cubes, dtype=np.float32)
    out = np.empty_like(cubes)
    for n in range(cubes.shape[0]):
        for c in range(cubes.shape[1]):
            out[n, c] = asinh_normalise(cubes[n, c]) if np.any(cubes[n, c]) else 0.0
    return out


def gaussian_psf(fwhm_px: float, size: int = 25):
    ax = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    s = fwhm_px / 2.3548
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s))
    return g / g.sum()


def fwhm_halfmax(psf):
    """FWHM (pixels) from the area above half maximum: A = pi FWHM^2 / 4."""
    psf = np.asarray(psf, float)
    return float(2.0 * np.sqrt((psf >= 0.5 * psf.max()).sum() / np.pi))


def fwhm_fit(psf, half=7):
    """FWHM (pixels) of a source at the stamp centre from a 2-D Gaussian fit - smooth and not quantised to
    whole pixels like the half-maximum area. Robust for single-exposure stamps: only the central
    (2*half+1)^2 pixels are fitted (distant neighbours / cosmic-ray remnants cannot capture the fit), the
    fit starts at the brightest pixel within 2 px of the centre, and sigma (0.2-8 px) and the centre
    (+-2 px) are bounded, so one axis cannot collapse to zero. Falls back to the half-maximum area
    measured on a 4x upsampled image."""
    psf = np.asarray(psf, float)
    n0 = psf.shape[0]
    c0 = n0 // 2
    h = min(half, c0)
    img = np.nan_to_num(psf[c0 - h:c0 + h + 1, c0 - h:c0 + h + 1])
    n = img.shape[0]
    c = (n - 1) / 2
    try:
        import warnings
        from astropy.modeling import fitting, models
        yy, xx = np.mgrid[:n, :n]
        near = np.hypot(yy - c, xx - c) <= 2
        py, px = np.unravel_index(np.argmax(np.where(near, img, -np.inf)), img.shape)
        if not img[py, px] > 0:
            raise ValueError("no positive peak near the centre")
        img = img / img[py, px]                   # amplitude ~1: e-/s values of ~0.01 stall the optimiser
        g0 = models.Gaussian2D(1.0, px, py, 1.0, 1.0,
                               bounds={"x_stddev": (0.2, 8.0), "y_stddev": (0.2, 8.0),
                                       "x_mean": (c - 2, c + 2), "y_mean": (c - 2, c + 2), "amplitude": (0, None)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = fitting.TRFLSQFitter()(g0, xx, yy, img)
        f = 2.3548 * np.sqrt(abs(g.x_stddev.value * g.y_stddev.value))
        if 0.5 < f < 2.3548 * 7.9:
            return float(f)
    except Exception:
        pass
    from scipy.ndimage import zoom
    up = zoom(np.clip(img, 0, None), 4, order=1)
    return fwhm_halfmax(up) / 4.0


def half_light_radius(img):
    """Radius (px) enclosing half the positive flux, about the image centre."""
    img = np.clip(np.asarray(img, float), 0, None)
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2).ravel()
    o = np.argsort(r)
    cog = np.cumsum(img.ravel()[o])
    if cog[-1] <= 0:
        return np.nan
    return float(r[o][np.searchsorted(cog, 0.5 * cog[-1])])


def cutout_snr(img, core_r=4, sky_r=40):
    """Peak SNR of a stamp: core mean over the MAD noise of the outer annulus."""
    img = np.asarray(img, float)
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    sky = img[r > sky_r]
    sigma = 1.4826 * np.median(np.abs(sky - np.median(sky))) + 1e-12
    return float((img[r <= core_r].mean() - np.median(sky)) / sigma)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------- sky matching
def unit_vectors(ra, dec):
    ra = np.deg2rad(np.asarray(ra, float))
    dec = np.deg2rad(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def sky_crossmatch(ra1, dec1, ra2, dec2, radius_arcsec=0.5):
    """Nearest neighbour of each catalogue-1 row in catalogue 2 (exact great-circle
    separations via a KD-tree on unit vectors). Returns idx2 (-1 = none), sep", matched."""
    n = len(ra1)
    if n == 0 or len(ra2) == 0:
        return np.full(n, -1), np.full(n, np.inf), np.zeros(n, bool)
    chord, idx = cKDTree(unit_vectors(ra2, dec2)).query(unit_vectors(ra1, dec1))
    sep = np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1))) * 3600.0
    hit = sep <= radius_arcsec
    return np.where(hit, idx, -1), sep, hit


def estimate_offset(ra1, dec1, ra2, dec2, search_arcsec=3.0, bin_arcsec=0.1, min_pairs=5):
    """Image-minus-catalogue offset (dRA cos dec, dDec) in arcsec, or None.
    Every image-catalogue pair closer than `search_arcsec` votes in a 2-D histogram of its offset; the PEAK is the
    common offset (random pairs spread evenly, true pairs pile up), so offsets larger than the final match radius
    are found too. The peak is refined by the median of the pairs within two bins of it. None when the peak holds
    fewer than `min_pairs` pairs or does not stand out (> 5x the mean bin)."""
    ra1, dec1 = np.asarray(ra1, float), np.asarray(dec1, float)
    ra2, dec2 = np.asarray(ra2, float), np.asarray(dec2, float)
    if len(ra1) == 0 or len(ra2) == 0:
        return None
    r0, d0 = np.median(ra1), np.median(dec1)
    c0 = np.cos(np.deg2rad(d0))
    a = np.c_[(ra1 - r0) * c0 * 3600, (dec1 - d0) * 3600]
    b = np.c_[(ra2 - r0) * c0 * 3600, (dec2 - d0) * 3600]
    lo, hi = a.min(0) - search_arcsec, a.max(0) + search_arcsec
    b = b[np.all((b >= lo) & (b <= hi), axis=1)]
    if len(b) == 0:
        return None
    pairs = cKDTree(a).query_ball_tree(cKDTree(b), search_arcsec)
    ii = np.repeat(np.arange(len(a)), [len(p) for p in pairs])
    jj = np.fromiter((j for p in pairs for j in p), int, count=len(ii))
    if len(ii) < min_pairs:
        return None
    d = a[ii] - b[jj]
    nb = int(round(2 * search_arcsec / bin_arcsec))
    H, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=nb, range=[[-search_arcsec, search_arcsec]] * 2)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    if H[i, j] < max(min_pairs, 5 * H.mean()):
        return None
    px, py = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    near = np.hypot(d[:, 0] - px, d[:, 1] - py) <= 2 * bin_arcsec
    return float(np.median(d[near, 0])), float(np.median(d[near, 1])), int(near.sum())


def match_with_offset(ra1, dec1, ra2, dec2, radius_arcsec=0.5, search_arcsec=3.0, estimate=True, groups=None):
    """Match image objects (1) to a catalogue (2) after removing the catalogue-to-image offset.
    The offset is measured SEPARATELY for every group of image objects (groups = their pointing): pointings come
    from different HST programmes whose astrometry differs by up to ~1", so one offset for all of them leaves the
    worst pointing unmatched. A group whose offset cannot be measured uses the offset of all objects together.
    Matches are one-to-one: if two image objects claim one catalogue row, only the closer keeps it.
    Returns idx2 (-1 = none), sep", hit, offset: (dra", ddec") without groups, {group: (dra", ddec")} with groups."""
    ra1, dec1 = np.asarray(ra1, float), np.asarray(dec1, float)
    ra2, dec2 = np.asarray(ra2, float), np.asarray(dec2, float)
    n = len(ra1)
    grp = np.zeros(n, int) if groups is None else np.asarray(groups)
    glob = estimate_offset(ra1, dec1, ra2, dec2, search_arcsec) if estimate and n else None
    idx, sep, hit = np.full(n, -1), np.full(n, np.inf), np.zeros(n, bool)
    offs = {}
    for g in pd.unique(grp):
        m = grp == g
        off = (estimate_offset(ra1[m], dec1[m], ra2, dec2, search_arcsec) if estimate and groups is not None else None) \
            or glob or (0.0, 0.0, 0)
        r2 = ra2 + off[0] / 3600 / np.cos(np.deg2rad(dec2))
        d2 = dec2 + off[1] / 3600
        i, s, h = sky_crossmatch(ra1[m], dec1[m], r2, d2, radius_arcsec)
        idx[m], sep[m], hit[m] = i, s, h
        offs[g if groups is not None else 0] = (round(off[0], 3), round(off[1], 3))
    order = np.argsort(np.where(hit, sep, np.inf))
    taken = set()
    for i in order:
        if hit[i]:
            if idx[i] in taken:
                hit[i] = False
            else:
                taken.add(idx[i])
    idx = np.where(hit, idx, -1)
    return idx, sep, hit, (offs if groups is not None else offs[0])


def fmt_offsets(off):
    """Log text of the per-pointing offsets."""
    if isinstance(off, dict):
        return ", ".join(f"{k} ({v[0]:+.2f}\", {v[1]:+.2f}\")" for k, v in off.items())
    return f"dRA {off[0]:+.3f}\", dDec {off[1]:+.3f}\""


# ---------------------------------------------------------------- metrics
def predictive_entropy(P):
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    return -(P * np.log(P)).sum(1) / np.log(P.shape[1])


def expected_calibration_error(P, y, n_bins=15):
    P, y = np.asarray(P), np.asarray(y)
    conf, correct = P.max(1), (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(metric_fn, y, P, n=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric(y, P) - small test sets need error bars."""
    rng = np.random.default_rng(seed)
    y, P = np.asarray(y), np.asarray(P)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(metric_fn(y[i], P[i]))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    vals = np.asarray(vals, float)
    if not np.isfinite(vals).any():
        return (np.nan, np.nan)
    return (float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------- identity
def iau_designation(ra, dec, prefix="P2P J"):
    """IAU style: coordinates TRUNCATED, e.g. P2P J123456.7+621234."""
    ra_h = (np.asarray(ra, float) / 15.0) % 24
    h = np.floor(ra_h)
    m = np.floor((ra_h - h) * 60)
    s = np.floor((((ra_h - h) * 60 - m) * 60) * 10) / 10
    d = np.abs(np.asarray(dec, float))
    dd = np.floor(d)
    dm = np.floor((d - dd) * 60)
    ds = np.floor(((d - dd) * 60 - dm) * 60)
    sign = np.where(np.asarray(dec) >= 0, "+", "-")
    return [f"{prefix}{int(a):02d}{int(b):02d}{c:04.1f}{sg}{int(e):02d}{int(f):02d}{int(g):02d}"
            for a, b, c, sg, e, f, g in zip(h, m, s, sign, dd, dm, ds)]


def noise_evidence(df: pd.DataFrame, cfg_baseline: dict) -> np.ndarray:
    """Physical evidence that a detection is spurious: very low SNR, a profile sharper
    than the PSF (no real source can be), or a single-band detection. The single-band test is the
    S/N-based cr_candidate of Stage 1c (no other covering band sees the source at S/N >= confirm_snr);
    the older flux-ratio test (band_consistency < single_band_max) is used only when that column is
    missing - a ratio test calls a real red galaxy that is faint in a shallow blue band 'single-band'."""
    ev = (df["snr"].to_numpy(float) < cfg_baseline["noise_snr_max"]) | \
         (df["fwhm_over_psf"].to_numpy(float) < cfg_baseline["sub_psf_factor"])
    if "cr_candidate" in df:
        ev |= as_bool(df["cr_candidate"]).to_numpy()
    elif "band_consistency" in df:
        bc = df["band_consistency"].to_numpy(float)
        ev |= np.isfinite(bc) & (bc < cfg_baseline.get("single_band_max", 0.05))
    return ev


def as_bool(series) -> pd.Series:
    """CSV round-trips turn booleans into strings / NaN; normalise them."""
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)


def galaxy_track(df) -> "np.ndarray":
    """The galaxy track: Galaxy, not an extreme outlier, and not a fragment of a brighter galaxy (Stage 2e
    fragment_of; 31 Oct). Every stage that selects galaxies uses this one rule."""
    cls = df["cluster_semantic"] if "cluster_semantic" in df else df["classification"]
    m = (cls == "Galaxy") & (df["anomaly_flag"].fillna(0) == 0)
    if "fragment_of" in df:
        m &= df["fragment_of"].isna()
    return m.to_numpy()
