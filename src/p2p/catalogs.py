"""Loading the external catalogues (GZH, Gaia DR3, CANDELS) into a standard form.

Each loader returns a DataFrame with *standard* column names (ra, dec, smooth,
...) taken from the mapping in config.yaml, cut to the sky area of the
pointings so large tables (GZH has ~114k rows) stay small in memory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .context import Ctx, StageSkipped

PATTERNS = {"gzh": ["*gz_hubble*", "*gzh*", "*galaxy_zoo*"],
            "gaia": ["*gaia*"],
            "candels": ["*kartaltepe*", "*candels*", "*k15*"]}


def find_catalog(ctx: Ctx, kind: str) -> Path | None:
    spec = ctx.cfg["catalogs"][kind].get("file", "auto")
    if spec and spec != "auto":
        p = ctx.root / spec if not Path(spec).is_absolute() else Path(spec)
        return p if p.exists() else None
    if not ctx.catalogs.exists():
        return None
    for pat in PATTERNS[kind]:
        hits = sorted(ctx.catalogs.glob(pat)) + sorted(ctx.catalogs.glob(f"*/{pat}"))     # also one folder down
        hits = [h for h in hits if h.is_file() and h.suffix.lower() in (".csv", ".gz", ".fits", ".fit", ".vot", ".xml", ".tsv")]
        if kind == "candels":                     # VizieR J/ApJS/221/11: table3 holds the class fractions
            hits = sorted(hits, key=lambda h: "table3" not in h.name.lower())
        if hits:
            return hits[0]
    return None


def _candidates(col) -> list[str]:
    """A mapping value may be one column name or a list of alternatives (first one found wins)."""
    return [str(c) for c in col] if isinstance(col, (list, tuple)) else [str(col)]


def _text_format(path: Path):
    """Separator and comment character of a text table. VizieR / Gaia-archive downloads may use ';' or
    tabs and '#' comment lines; GZH is a plain comma-separated file."""
    import gzip
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        first = ""
        for line in f:
            if line.strip() and not line.startswith("#"):
                first = line
                break
    sep = max([",", ";", "\t", "|"], key=first.count) if first else ","
    return (sep if first.count(sep) else ","), "#"


def read_table(path: Path, usecols=None) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.table import Table
        t = Table.read(path)
        names = [c for c in t.colnames if len(t[c].shape) <= 1]
        df = t[names].to_pandas()
        return df[[c for c in df.columns if c in usecols]] if usecols else df
    if name.endswith((".vot", ".xml", ".vot.gz")):
        from astropy.table import Table
        return Table.read(path, format="votable").to_pandas()
    sep, comment = _text_format(path)
    header = pd.read_csv(path, nrows=0, sep=sep, comment=comment).columns
    cols = [c for c in header if usecols is None or c in usecols]
    return pd.read_csv(path, usecols=cols, sep=sep, comment=comment, low_memory=False)


def read_mapped(path: Path, mapping: dict) -> pd.DataFrame:
    """Read only the mapped columns (case-insensitive) - keeps big tables light."""
    want = {c.lower() for v in mapping.values() for c in _candidates(v)}
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".dat")):
        sep, comment = _text_format(path)
        header = pd.read_csv(path, nrows=0, sep=sep, comment=comment).columns
        cols = [c for c in header if c.strip().lower() in want]
        df = pd.read_csv(path, usecols=cols, sep=sep, comment=comment, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    df = read_table(path)
    return df[[c for c in df.columns if c.lower() in want]]


def _to_degrees(values, is_ra):
    """Numbers stay numbers; sexagesimal strings ('03:32:24.6' / '-27:48:31') are converted to degrees."""
    num = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip()
    sexa = num.isna() & text.str.contains(r"^[+-]?\d+[: ]\d+[: ]\d", regex=True)
    if sexa.any():
        from astropy.coordinates import Angle
        import astropy.units as u
        unit = u.hourangle if is_ra else u.deg
        num = num.copy()
        num[sexa] = [Angle(t, unit=unit).degree for t in text[sexa]]
    return num


def standardise(df: pd.DataFrame, mapping: dict, kind: str, required=("ra", "dec")) -> pd.DataFrame:
    """Rename catalogue columns to standard names. Matching is case-insensitive and a mapping value
    may list alternatives. A missing REQUIRED column stops the run with the closest names; a missing
    optional one is filled with NaN. Rows without a usable position are dropped."""
    import difflib
    lower = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    missing = []
    for std, col in mapping.items():
        src = next((lower[c.lower()] for c in _candidates(col) if c.lower() in lower), None)
        if src is None:
            missing.append((std, col))
            out[std] = np.nan
        elif std in ("ra", "dec"):
            out[std] = _to_degrees(df[src], is_ra=(std == "ra"))
        elif std in ("source_id", "id"):
            out[std] = df[src]
        else:
            out[std] = pd.to_numeric(df[src], errors="coerce")
    hard = [m for m in missing if m[0] in required]
    if hard:
        hints = {std: difflib.get_close_matches(str(_candidates(col)[0]), list(df.columns), n=3, cutoff=0.3)
                 for std, col in hard}
        raise ValueError(f"{kind}: required column(s) not found: {hard}. Closest names in the file: {hints}.\n"
                         f"Fix 'catalogs.{kind}.columns' in config.yaml (python tools/check_catalog_columns.py "
                         "lists every column).")
    if missing:
        print(f"[catalogs] {kind}: optional column(s) missing, filled with NaN: {missing}")
    bad = out["ra"].isna() | out["dec"].isna()
    if bad.mean() > 0.5:
        raise ValueError(f"{kind}: {bad.mean():.0%} of the rows have no readable RA/Dec - check the file format.")
    return out[~bad]


def sky_box(metas, margin_deg=0.02):
    pts = np.vstack([np.asarray(m["footprint"]) for m in metas])
    return (pts[:, 0].min() - margin_deg, pts[:, 0].max() + margin_deg,
            pts[:, 1].min() - margin_deg, pts[:, 1].max() + margin_deg)


def cut_to_box(df, box):
    r0, r1, d0, d1 = box
    return df[(df.ra >= r0) & (df.ra <= r1) & (df.dec >= d0) & (df.dec <= d1)].reset_index(drop=True)


# ---------------------------------------------------------------- GZH
def load_gzh(ctx: Ctx, metas, required=True) -> pd.DataFrame | None:
    path = find_catalog(ctx, "gzh")
    if path is None:
        if required:
            raise StageSkipped("No Galaxy Zoo: Hubble table in data/catalogs (e.g. gz_hubble_main.csv.gz). "
                               "See README section 'Catalogues'.")
        return None
    mapping = ctx.cfg["catalogs"]["gzh"]["columns"]
    df = standardise(read_mapped(path, mapping), mapping, "gzh")
    df["gzh_row"] = df.index.to_numpy()          # row number in the original GZH file
    return cut_to_box(df, sky_box(metas))


# ---------------------------------------------------------------- Gaia
def load_gaia(ctx: Ctx, metas) -> pd.DataFrame | None:
    cfg = ctx.cfg["catalogs"]["gaia"]
    path = find_catalog(ctx, "gaia")
    if path is not None:
        df = standardise(read_mapped(path, cfg["columns"]), cfg["columns"], "gaia")
        return cut_to_box(df, sky_box(metas))
    if cfg.get("query_online"):
        return query_gaia_online(ctx, metas)
    return None


def query_gaia_online(ctx: Ctx, metas) -> pd.DataFrame | None:
    try:
        from astroquery.gaia import Gaia
    except Exception:
        print("[catalogs] astroquery not installed - cannot query Gaia online.")
        return None
    r0, r1, d0, d1 = sky_box(metas)
    q = ("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, teff_gspphot "
         f"FROM gaiadr3.gaia_source WHERE ra BETWEEN {r0} AND {r1} AND dec BETWEEN {d0} AND {d1}")
    try:
        t = Gaia.launch_job_async(q).get_results().to_pandas()
    except Exception as e:
        print(f"[catalogs] Gaia query failed: {e}")
        return None
    out = ctx.catalogs / "gaia_dr3_query.csv"
    ctx.catalogs.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    return standardise(t, ctx.cfg["catalogs"]["gaia"]["columns"], "gaia")


def propagate_gaia(gaia: pd.DataFrame, to_epoch: float, from_epoch: float = 2016.0) -> pd.DataFrame:
    """Move Gaia positions to the HST observation epoch with proper motions."""
    g = gaia.copy()
    if not np.isfinite(to_epoch):
        return g
    dt = to_epoch - from_epoch
    pmra = g["pmra"].fillna(0).to_numpy(float)          # mas/yr, includes cos(dec)
    pmdec = g["pmdec"].fillna(0).to_numpy(float)
    cosd = np.cos(np.deg2rad(g["dec"].to_numpy(float)))
    g["ra"] = g["ra"] + pmra * dt / 3.6e6 / np.maximum(cosd, 1e-6)
    g["dec"] = g["dec"] + pmdec * dt / 3.6e6
    return g


# ---------------------------------------------------------------- CANDELS
K15_CLASS_COLUMNS = ("spheroid", "disk", "irregular", "point_source", "unclassifiable")


def load_candels(ctx: Ctx, metas) -> pd.DataFrame | None:
    """Kartaltepe et al. (2015) FRACTIONAL table (VizieR J/ApJS/221/11/table3). The five class fractions
    are required: if they were silently NaN every galaxy would become 'spheroid'. The same galaxy can
    appear at two depths; the row with the most classifiers is kept."""
    path = find_catalog(ctx, "candels")
    if path is None:
        return None
    mapping = ctx.cfg["catalogs"]["candels"]["columns"]
    df = standardise(read_mapped(path, mapping), mapping, "candels", required=("ra", "dec") + K15_CLASS_COLUMNS)
    if "n_classifiers" in df and df.n_classifiers.notna().any():
        key = (df.ra * 3600 / 0.2).round().astype("int64").astype(str) + "_" + \
              (df.dec * 3600 / 0.2).round().astype("int64").astype(str)
        before = len(df)
        df = df.assign(_k=key).sort_values("n_classifiers", ascending=False).drop_duplicates("_k").drop(columns="_k")
        if before > len(df):
            print(f"[catalogs] candels: {before - len(df)} duplicate rows (same galaxy, other depth) removed")
    return cut_to_box(df.reset_index(drop=True), sky_box(metas))
