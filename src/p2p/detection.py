"""Background modelling and source detection shared by Stage 1c and Stage 3c."""
from __future__ import annotations

import numpy as np
from photutils.background import Background2D, MedianBackground
from scipy import ndimage

from .compat import deblend, detect
from .utils import gaussian_psf


def subtract_background(sci, valid, box=64, filt=3):
    """2-D background (median estimator) fitted on covered pixels only, so chip
    gaps and frame edges (WHT = 0) do not drag the estimate. Returns the
    background-subtracted image (0 where invalid) and the background rms map."""
    data = np.where(valid, sci, 0.0).astype(np.float32)
    bkg = Background2D(data, box, filter_size=(filt, filt), coverage_mask=~valid,
                       bkg_estimator=MedianBackground(), fill_value=0.0)
    sub = np.where(valid, data - bkg.background, 0.0).astype(np.float32)
    rms = np.where(valid, bkg.background_rms, np.nan).astype(np.float32)
    rms = np.where(np.isfinite(rms) & (rms > 0), rms, np.nanmedian(rms))
    return sub, rms


def smooth(sub, fwhm_px=3.0):
    k = gaussian_psf(fwhm_px, size=int(2 * round(1.5 * fwhm_px) + 1))
    return ndimage.convolve(sub, k, mode="constant", cval=0.0)


def detect_and_deblend(sub, rms, valid, cfg, deblend_sources=True):
    """Threshold = nsigma x rms x correlated-noise factor on the smoothed image,
    then multi-threshold deblending (SExtractor-style)."""
    conv = smooth(sub, cfg["kernel_fwhm_px"])
    thr = cfg["nsigma"] * rms * cfg.get("correlated_noise_factor", 1.0)
    segm = detect(conv, thr, cfg["npixels"], mask=~valid)
    if segm is not None and deblend_sources:
        segm = deblend(conv, segm, cfg["npixels"], cfg["deblend_nlevels"], cfg["deblend_contrast"])
    return segm, conv
