"""photutils renamed several arguments and attributes between 1.x/2.x and 3.x
(npixels -> n_pixels, nlevels -> n_levels, xcentroid -> x_centroid, ...).
These wrappers accept either version so the project runs on both."""
from __future__ import annotations

import inspect

import numpy as np

from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources

_DEBLEND = inspect.signature(deblend_sources).parameters
_ALIASES = {"x_centroid": "xcentroid", "y_centroid": "ycentroid", "n_labels": "nlabels",
            "segment_area": "area", "semimajor_axis": "semimajor_sigma",
            "semiminor_axis": "semiminor_sigma", "kron_flux_err": "kron_fluxerr",
            "segment_flux_err": "segment_fluxerr"}


def detect(data, threshold, npixels, mask=None):
    return detect_sources(data, threshold, npixels, mask=mask)


def deblend(data, segm, npixels, nlevels=32, contrast=0.001):
    kw = {"contrast": contrast}
    kw["n_levels" if "n_levels" in _DEBLEND else "nlevels"] = nlevels
    if "progress_bar" in _DEBLEND:
        kw["progress_bar"] = False
    return deblend_sources(data, segm, npixels, **kw)


def attr(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    return getattr(obj, _ALIASES[name])


def values(obj, name):
    """Catalogue property as a plain float array (strips astropy units)."""
    v = attr(obj, name)
    v = getattr(v, "value", v)
    return np.asarray(v, dtype=float)


def n_labels(segm) -> int:
    if segm is None:
        return 0
    return int(attr(segm, "n_labels"))


__all__ = ["detect", "deblend", "attr", "values", "n_labels", "SourceCatalog"]
