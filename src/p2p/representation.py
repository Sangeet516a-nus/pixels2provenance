"""Stage 2a helpers: image normalisation for the representation autoencoder and its diagnostics.

Why a field-level normalisation (representation.normalisation: field_noise)
---------------------------------------------------------------------------
The original CAE input stretched every cutout band to its own 1st-99.5th percentile.
In single-exposure frames the 99.5th percentile is often a cosmic ray, so the apparent
background level changed from cutout to cutout, and the first two principal components
of the CAE latent tracked the mean level of F475W and F606W (Spearman rho 0.98 / 0.97)
instead of the source.  Here every band of every cutout is put on ONE scale per pointing:

    x' = (x - local background) / sigma_band(pointing)          [units of the background noise]
    y  = asinh(x' / a) / asinh(M / a)                             a = asinh_softening_sigma, M = asinh_max_sigma

* local background = median of the cutout's background pixels (segmentation code 0) in that band
* sigma_band       = 1.4826 x MAD of background pixels of up to 400 cutouts of the pointing
                     (an image-noise statistic: it uses no labels, so it cannot leak)
* pixels without data in a band (outside that band's footprint) are set to 0 and get a
  separate 0/1 "data present" channel, so a missing band is not mistaken for a dark source.

The loss weights (pixel_weights) make the autoencoder spend its capacity on the source:
target footprint (grown by 2 px) x target_weight, background 1, cosmic-ray pixels and
pixels without data 0.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SEG_BG, SEG_TARGET, SEG_NEIGHBOUR, SEG_CR = 0, 1, 2, 3


def band_sigmas(cut, seg, rows, max_stamps=400, seed=42):
    """Robust background sigma per band from the background pixels of up to `max_stamps` cutouts."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    pick = rows if len(rows) <= max_stamps else rng.choice(rows, max_stamps, replace=False)
    nb = cut.shape[1]
    vals = [[] for _ in range(nb)]
    for i in np.sort(pick):
        c = np.asarray(cut[i], np.float32)
        bg = np.asarray(seg[i]) == SEG_BG
        for b in range(nb):
            v = c[b][bg & (c[b] != 0)]
            if v.size > 50:
                vals[b].append(v - np.median(v))
    sig = []
    for b in range(nb):
        if vals[b]:
            v = np.concatenate(vals[b])
            sig.append(float(1.4826 * np.median(np.abs(v))) or 1.0)
        else:
            sig.append(np.nan)                                   # band never covered in this pointing
    return np.array(sig, np.float32)


def normalise_cutouts(cut, seg, idx, sigma_rows, a=3.0, m=1000.0):
    """Field-noise asinh normalisation of cutouts `idx`.

    sigma_rows: (len(idx), bands) background sigma for each cutout's pointing.
    Returns x (n, bands, S, S) float16 in about [-0.3, 1] and present (n, bands, S, S) bool."""
    idx = np.asarray(idx)
    n, nb, S = len(idx), cut.shape[1], cut.shape[-1]
    x = np.zeros((n, nb, S, S), np.float16)
    present = np.zeros((n, nb, S, S), bool)
    den = np.arcsinh(m / a)
    for k, i in enumerate(idx):
        c = np.asarray(cut[i], np.float32)
        bg = np.asarray(seg[i]) == SEG_BG
        for b in range(nb):
            p = c[b] != 0
            if not p.any() or not np.isfinite(sigma_rows[k, b]):
                continue
            ref = c[b][bg & p]
            lb = float(np.median(ref)) if ref.size > 50 else 0.0
            y = np.arcsinh((c[b] - lb) / (sigma_rows[k, b] * a)) / den
            x[k, b] = np.where(p, np.clip(y, -0.3, 1.0), 0.0)
            present[k, b] = p
    return x, present


def pixel_weights(seg, present, target_weight=10.0, ignore_cr=True, grow_px=2, chunk=512):
    """Loss weight per pixel and band: target footprint (grown) x target_weight, background 1,
    cosmic-ray pixels (segmentation code 3) and pixels without data 0. Normalised to mean 1 per cutout.
    seg (n, S, S), present (n, bands, S, S) -> float16 (n, bands, S, S); computed in chunks to save memory."""
    n = len(present)
    out = np.zeros(present.shape, np.float16)
    for s in range(0, n, chunk):
        sg = np.asarray(seg[s:s + chunk])
        w = np.ones(sg.shape, np.float32)
        tgt = sg == SEG_TARGET
        if grow_px > 0:
            tgt = np.stack([ndimage.binary_dilation(t, iterations=grow_px) for t in tgt])
        w[tgt] = float(target_weight)
        if ignore_cr:
            w[sg == SEG_CR] = 0.0
        w = w[:, None, :, :] * present[s:s + chunk].astype(np.float32)
        tot = w.sum(axis=(1, 2, 3), keepdims=True)
        out[s:s + chunk] = w * (w[0].size / np.maximum(tot, 1e-6))
    return out


# ------------------------------------------------------------------ diagnostics
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def band_levels(cut, idx, per_stamp=True):
    """Mean level of each band of each cutout after the ORIGINAL per-stamp asinh stretch: the nuisance
    quantity that dominated the old latent. Used only to check that the new latent does not track it."""
    from .utils import asinh_stack
    out = []
    for s in range(0, len(idx), 256):
        X = asinh_stack(np.asarray(cut[np.asarray(idx[s:s + 256])], np.float32))
        out.append(X.mean(axis=(2, 3)))
    return np.concatenate(out)
