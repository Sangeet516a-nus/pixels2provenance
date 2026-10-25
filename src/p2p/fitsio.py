"""Reading HST drizzled products (_drc / _drz) and grouping them into pointings."""
from __future__ import annotations

import fnmatch
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

warnings.simplefilter("ignore", FITSFixedWarning)


def find_products(raw_dir: Path, patterns) -> list[Path]:
    files = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and any(fnmatch.fnmatch(p.name.lower(), pat.lower()) for pat in patterns):
            files.append(p)
    # prefer _drc over _drz for the same rootname
    by_root = {}
    for f in files:
        root = f.name.lower().replace("_drc", "").replace("_drz", "")
        if root not in by_root or "_drc" in f.name.lower():
            by_root[root] = f
    return sorted(by_root.values())


def filter_name(prim, sci) -> str:
    for hdr in (prim, sci):
        for k in ("FILTER", "FILTER1", "FILTER2"):
            v = str(hdr.get(k, "")).strip().upper()
            if v and not v.startswith("CLEAR") and v not in ("N/A", "NONE"):
                return v
    return "UNKNOWN"


def instrument_key(prim) -> str:
    inst = str(prim.get("INSTRUME", "")).upper()
    det = str(prim.get("DETECTOR", "")).upper()
    if inst == "WFC3":
        return f"WFC3_{det}" if det else "WFC3"
    return inst or "UNKNOWN"


def read_meta(path: Path) -> dict:
    """Header-only read: everything later stages need to know about a file."""
    with fits.open(path, memmap=True) as hdul:
        names = [h.name for h in hdul]
        prim = hdul[0].header
        sci_ext = "SCI" if "SCI" in names else (1 if len(hdul) > 1 else 0)
        sci = hdul[sci_ext].header
        w = WCS(sci).celestial
        nx, ny = int(sci.get("NAXIS1", 0)), int(sci.get("NAXIS2", 0))
    ok_wcs = w.has_celestial and str(sci.get("CTYPE1", "")).startswith("RA")
    scale = float(proj_plane_pixel_scales(w)[0] * 3600) if ok_wcs else np.nan
    fp = w.calc_footprint(axes=(nx, ny)) if ok_wcs else np.zeros((4, 2))
    centre = w.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2) if ok_wcs else (np.nan, np.nan)
    photflam = sci.get("PHOTFLAM", prim.get("PHOTFLAM"))
    photplam = sci.get("PHOTPLAM", prim.get("PHOTPLAM"))
    zp = float(-2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408) if photflam and photplam else np.nan
    date = prim.get("DATE-OBS") or sci.get("DATE-OBS")
    try:
        epoch = float(Time(date).decimalyear) if date else np.nan
    except Exception:
        epoch = np.nan
    return {
        "file": str(path), "name": path.name,
        "product_type": "drc" if "_drc" in path.name.lower() else ("drz" if "_drz" in path.name.lower() else "other"),
        "has_sci": "SCI" in names, "has_wht": "WHT" in names, "has_ctx": "CTX" in names,
        "wcs_ok": bool(ok_wcs), "naxis1": nx, "naxis2": ny,
        "instrument": instrument_key(prim), "telescope": str(prim.get("TELESCOP", "")),
        "filter": filter_name(prim, sci), "targname": str(prim.get("TARGNAME", "")),
        "proposid": prim.get("PROPOSID"), "exptime": float(prim.get("EXPTIME", sci.get("EXPTIME", np.nan))),
        "ndrizim": prim.get("NDRIZIM", sci.get("NDRIZIM")), "date_obs": str(date), "epoch": epoch,
        "bunit": str(sci.get("BUNIT", "")), "photflam": photflam, "photplam": photplam, "zp_ab": zp,
        "pixscale_arcsec": scale, "ra_centre": float(centre[0]), "dec_centre": float(centre[1]),
        "footprint": np.asarray(fp).tolist(),
    }


def load_image(path: str | Path):
    """SCI (float32), WHT (float32 or None) and the celestial WCS."""
    with fits.open(path, memmap=False) as hdul:
        names = [h.name for h in hdul]
        sci_ext = "SCI" if "SCI" in names else 1
        sci = np.asarray(hdul[sci_ext].data, dtype=np.float32)
        wht = np.asarray(hdul["WHT"].data, dtype=np.float32) if "WHT" in names else None
        w = WCS(hdul[sci_ext].header).celestial
    return sci, wht, w


def load_ctx(path: str | Path):
    """Drizzle context map (CTX): bit k set = input image k contributes to the pixel. None if absent
    or 3-D (more than 32 inputs)."""
    with fits.open(path, memmap=False) as hdul:
        names = [h.name for h in hdul]
        if "CTX" not in names or hdul["CTX"].data is None or hdul["CTX"].data.ndim != 2:
            return None
        return np.asarray(hdul["CTX"].data).astype(np.int64)


def valid_mask(sci, wht):
    m = np.isfinite(sci)
    if wht is not None:
        m &= wht > 0
    return m


def is_external(meta: dict, patterns) -> bool:
    text = (meta["name"] + " " + meta.get("targname", "")).lower()
    return any(p.lower() in text for p in patterns or [])


def _overlap_fraction(a: dict, b: dict) -> float:
    """Overlap of two footprints' RA/Dec boxes as a fraction of the SMALLER box (cos(dec)-corrected)."""
    pa, pb = np.asarray(a["footprint"]), np.asarray(b["footprint"])
    cosd = np.cos(np.deg2rad(0.5 * (pa[:, 1].mean() + pb[:, 1].mean())))
    w = min(pa[:, 0].max(), pb[:, 0].max()) - max(pa[:, 0].min(), pb[:, 0].min())
    h = min(pa[:, 1].max(), pb[:, 1].max()) - max(pa[:, 1].min(), pb[:, 1].min())
    if w <= 0 or h <= 0:
        return 0.0
    area = lambda p: np.ptp(p[:, 0]) * cosd * np.ptp(p[:, 1])
    return float(w * cosd * h / max(min(area(pa), area(pb)), 1e-12))


def group_pointings(metas: list[dict], radius_arcsec: float, min_overlap: float = 0.5) -> list[list[int]]:
    """Union-find: files whose centres are closer than radius_arcsec, OR whose footprints overlap by at
    least min_overlap of the smaller one, form one pointing. The overlap rule matters when the bands of
    one field come from different HST programmes, with centres far apart but the same sky."""
    n = len(metas)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ra = np.array([m["ra_centre"] for m in metas])
    dec = np.array([m["dec_centre"] for m in metas])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot((ra[i] - ra[j]) * np.cos(np.deg2rad(dec[i])), dec[i] - dec[j]) * 3600
            if d < radius_arcsec or _overlap_fraction(metas[i], metas[j]) >= min_overlap:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # stable order: by RA of the group centre
    out = list(groups.values())
    out.sort(key=lambda g: (np.mean(ra[g]), np.mean(dec[g])))
    return out


def bbox_of(metas_in_group: list[dict]):
    pts = np.vstack([np.asarray(m["footprint"]) for m in metas_in_group])
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())
