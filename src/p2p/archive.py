"""Find and download the best HST images for a sky position (MAST), and the Gaia DR3 cone around it.

Which file is "best" for this pipeline, per filter:
  1. it must CONTAIN the position (the observation footprint s_region, not just a nearby centre);
  2. a COMBINED product (several exposures drizzled together, cosmic rays rejected) beats a single exposure:
     HAP visit-level filter products (hst_<prop>_<visit>_<inst>_<det>_<filter>_<6 chars>) and classic association
     products (<9 chars> ending in digits) are combined; HAP exposure-level products (8-character suffix) are
     single exposures. Single exposures are the main error source of this pipeline (cosmic rays).
  3. then the longest total exposure time, then the position closest to the image centre.
Multi-visit mosaics (skycells) are left out by default: they are several GB.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# one band per colour, bluest first; the first available filter of each group is taken in 'auto'
FILTER_GROUPS = {
    "ACS/WFC": [["F475W", "F435W", "F555W"], ["F606W", "F625W"], ["F814W", "F775W", "F850LP"]],
    "WFC3/UVIS": [["F475W", "F438W", "F390W"], ["F606W", "F555W"], ["F814W", "F775W"]],
    "WFC3/IR": [["F105W", "F110W"], ["F125W", "F140W"], ["F160W"]],
}


@dataclass
class Choice:
    filter: str
    obs_id: str
    filename: str
    combined: bool
    exptime_s: float
    offset_arcmin: float
    size_mb: float
    target_name: str
    proposal_id: str


def product_kind(obs_id: str) -> str:
    """'hap-visit' (combined), 'hap-exposure' (single), 'classic-assoc' (combined), 'classic-exp' (single), 'mosaic'."""
    o = obs_id.lower()
    if "skycell" in o:
        return "mosaic"
    if o.startswith("hst_"):
        last = o.split("_")[-1]
        return "hap-visit" if len(last) == 6 else "hap-exposure"
    if re.fullmatch(r"[a-z0-9]{6}\d{3}", o):
        return "classic-assoc"
    return "classic-exp"


def is_combined(obs_id: str) -> bool:
    return product_kind(obs_id) in ("hap-visit", "classic-assoc", "mosaic")


def _polygons(s_region: str):
    """All polygons of an s_region string ('POLYGON [ICRS] ra dec ra dec ...', possibly several)."""
    out = []
    for part in re.split(r"(?i)polygon", str(s_region))[1:]:
        nums = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", part)]
        if len(nums) >= 6:
            out.append(np.array(nums[: len(nums) // 2 * 2]).reshape(-1, 2))
    return out


def contains(s_region: str, ra: float, dec: float) -> bool:
    from matplotlib.path import Path as MPath
    for poly in _polygons(s_region):
        p = poly.copy()
        p[:, 0] = (p[:, 0] - ra + 180) % 360 - 180          # unwrap RA around the target
        p[:, 0] *= np.cos(np.radians(dec))
        p[:, 1] -= dec
        if MPath(p).contains_point((0.0, 0.0)):
            return True
    return False


def rank_observations(obs, ra: float, dec: float, allow_single=True, allow_mosaic=False):
    """obs: astropy Table / pandas DataFrame from Observations.query_criteria. Returns a DataFrame sorted best first,
    with columns kind, combined, contains, offset_arcmin."""
    import pandas as pd
    df = obs.to_pandas() if hasattr(obs, "to_pandas") else pd.DataFrame(obs)
    if df.empty:
        return df
    df["kind"] = df.obs_id.map(product_kind)
    df["combined"] = df.obs_id.map(is_combined)
    df["contains"] = [contains(s, ra, dec) for s in df.s_region]
    cosd = np.cos(np.radians(dec))
    df["offset_arcmin"] = 60 * np.hypot((df.s_ra - ra) * cosd, df.s_dec - dec)
    df = df[df.contains]
    if not allow_mosaic:
        df = df[df.kind != "mosaic"]
    if not allow_single:
        df = df[df.combined]
    return df.sort_values(["combined", "t_exptime", "offset_arcmin"], ascending=[False, False, True])


def overlap_fraction(region_a: str, region_b: str, ra: float, dec: float, n: int = 30) -> float:
    """Share of footprint A that footprint B also covers (grid sampling; fine for choosing files)."""
    from matplotlib.path import Path as MPath
    cosd = np.cos(np.radians(dec))
    def local(poly):
        p = poly.copy()
        p[:, 0] = ((p[:, 0] - ra + 180) % 360 - 180) * cosd
        return p
    A = [local(p) for p in _polygons(region_a)]
    B = [MPath(local(p)) for p in _polygons(region_b)]
    if not A or not B:
        return 0.0
    pts = np.vstack([a for a in A])
    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), n)
    ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n)
    grid = np.array([(x, y) for x in xs for y in ys])
    in_a = np.any([MPath(a).contains_points(grid) for a in A], axis=0)
    if not in_a.any():
        return 0.0
    in_b = np.any([b.contains_points(grid) for b in B], axis=0)
    return float((in_a & in_b).sum() / in_a.sum())


def _filters_of(v) -> set:
    return {x.strip().upper() for x in str(v).replace(",", ";").split(";") if x.strip()}


def search(ra: float, dec: float, filters="auto", instrument="ACS/WFC", radius_deg=0.03,
           allow_single=True, allow_mosaic=False, log=print) -> list[Choice]:
    """One Choice per filter (or per colour group when filters == 'auto'). The reddest group (the detection band)
    is chosen first; the other bands then prefer files that overlap it by >= 50 %, so Stage 1a groups all bands
    into ONE pointing."""
    from astroquery.mast import Observations
    groups = FILTER_GROUPS.get(instrument, FILTER_GROUPS["ACS/WFC"]) if filters == "auto" else [[f] for f in filters]
    flat = sorted({f.upper() for g in groups for f in g})
    obs = Observations.query_criteria(coordinates=f"{ra} {dec}", radius=f"{radius_deg} deg", obs_collection="HST",
                                      dataproduct_type="image", calib_level=3, instrument_name=instrument,
                                      filters=[f"*{f}*" for f in flat])   # ACS lists e.g. "F814W;CLEAR2L"
    log(f"[archive] ra={ra} dec={dec}: {len(obs)} {instrument} observations in {flat}")
    ranked = rank_observations(obs, ra, dec, allow_single, allow_mosaic)
    picked: dict[int, tuple] = {}
    ref_region = None
    for gi in sorted(range(len(groups)), key=lambda i: -_wavelength_nm(groups[i][0])):
        for f in groups[gi]:
            cand = ranked[[f.upper() in _filters_of(v) for v in ranked.filters]] if len(ranked) else ranked
            if cand.empty:
                continue
            if ref_region is not None:
                cand = cand.assign(ov=[overlap_fraction(ref_region, r, ra, dec) >= 0.5 for r in cand.s_region])
                cand = cand.sort_values(["ov", "combined", "t_exptime", "offset_arcmin"],
                                        ascending=[False, False, False, True])
            hit = _first_product(Observations, cand, f)
            if hit:
                picked[gi] = hit
                if ref_region is None:
                    ref_region = hit[1]
                break
    choices = [picked[i][0] for i in sorted(picked)]
    for c in choices:
        log(f"[archive]   {c.filter:7s} {c.filename:55s} {'combined' if c.combined else 'SINGLE exposure':16s} "
            f"{c.exptime_s:7.0f} s  {c.offset_arcmin:4.1f}' from centre  {c.size_mb:6.0f} MB")
    missing = [g[0] for i, g in enumerate(groups) if i not in picked]
    if missing:
        log(f"[archive]   no covering {instrument} image for: {missing}")
    if any(not c.combined for c in choices):
        log("[archive]   WARNING: single-exposure files carry every cosmic ray - expect many spurious detections")
    return choices


def _wavelength_nm(band: str) -> float:
    m = re.search(r"(\d+)", band)
    v = float(m.group(1)) if m else 0.0
    return v * 10 if v < 200 else v


def _first_product(Observations, cand, f):
    for _, row in cand.iterrows():
        prods = Observations.filter_products(Observations.get_product_list(str(row.obsid)),
                                             productType="SCIENCE", extension="fits").to_pandas()
        for sub in ("DRC", "DRZ"):
            hit = prods[(prods.productSubGroupDescription == sub) &
                        prods.productFilename.str.lower().str.startswith(str(row.obs_id).lower())]
            if len(hit):
                r = hit.iloc[0]
                return (Choice(f.upper(), str(row.obs_id), str(r.productFilename), bool(row.combined),
                               float(row.t_exptime), round(float(row.offset_arcmin), 2),
                               round(float(r["size"]) / 1e6, 1), str(row.target_name), str(row.proposal_id)),
                        str(row.s_region))
    return None


def download(choices: list[Choice], out_dir, log=print) -> list[Path]:
    from astroquery.mast import Observations
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in choices:
        dest = out_dir / c.filename
        if dest.exists():
            log(f"[archive] already there: {dest.name}")
            paths.append(dest)
            continue
        uri = f"mast:HST/product/{c.filename}"
        log(f"[archive] downloading {c.filename} ({c.size_mb:.0f} MB) ...")
        status, msg, _ = Observations.download_file(uri, local_path=str(dest))
        if status != "COMPLETE":
            raise RuntimeError(f"download of {c.filename} failed: {status} {msg}")
        paths.append(dest)
    return paths


def gaia_cone(ra: float, dec: float, radius_deg: float, out_dir, log=print) -> Path:
    from astroquery.gaia import Gaia
    out = Path(out_dir) / f"gaia_dr3_{ra:.3f}_{dec:+.3f}.csv"
    if out.exists():
        log(f"[gaia] already there: {out.name}")
        return out
    q = ("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, teff_gspphot FROM gaiadr3.gaia_source "
         f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))")
    t = Gaia.launch_job(q).get_results()
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_pandas().to_csv(out, index=False)
    log(f"[gaia] {len(t)} sources within {radius_deg} deg -> {out.name}")
    return out


def choices_table(choices: list[Choice]):
    import pandas as pd
    return pd.DataFrame([asdict(c) for c in choices])
