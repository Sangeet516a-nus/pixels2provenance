"""From sky positions to a finished run: search MAST, download, choose train/apply mode, run the pipeline.

Used by tools/run_field.py (command line).

    from p2p.field import Target, plan, prepare, run_pipeline_stream
    targets = [Target(53.12, -27.80, "GOODS-S")]
    p = plan(root, targets)                  # search only: what would be downloaded, GZH / CANDELS coverage
    cfg_file = prepare(root, p, fresh=True)  # archive the old run, download, write config_run.yaml
    for line in run_pipeline_stream(root, cfg_file):
        print(line, end="")
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import archive

GZH_URL = "https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-hubble/gz_hubble_main.csv.gz"
CANDELS_URL = ("https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
               + urllib.parse.quote('SELECT * FROM "J/ApJS/221/11/table3"'))


@dataclass
class Target:
    ra: float
    dec: float
    name: str = ""


def parse_coord(ra, dec) -> tuple[float, float]:
    """Degrees ('53.12', '-27.80') or sexagesimal ('03:32:28.8', '-27:48:00' / '3h32m28.8s', '-27d48m')."""
    ra, dec = str(ra).strip(), str(dec).strip()
    if any(ch in ra for ch in ": hHmM") or any(ch in dec for ch in ": dDmM'\""):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        return float(c.ra.deg), float(c.dec.deg)
    return float(ra), float(dec)


def parse_targets(text: str) -> list:
    """One target per line: 'RA, Dec[, name]' (commas or tabs)."""
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        ra, dec = parse_coord(parts[0], parts[1])
        out.append(Target(ra, dec, parts[2] if len(parts) > 2 else ""))
    return out


@dataclass
class Plan:
    targets: list
    instrument: str = "ACS/WFC"
    filters: object = "auto"
    choices: dict = field(default_factory=dict)      # target index -> list[archive.Choice as dict]
    gzh_counts: dict = field(default_factory=dict)   # target index -> GZH galaxies within the footprint radius
    candels_counts: dict = field(default_factory=dict)
    mode: str = "train"
    reason: str = ""

    def to_json(self):
        d = asdict(self)
        d["targets"] = [asdict(t) if not isinstance(t, dict) else t for t in self.targets]
        return d


def load_cfg(root: Path, config="config.yaml") -> dict:
    return yaml.safe_load((Path(root) / config).read_text())


# ------------------------------------------------------------------ reference catalogues
def ensure_catalogues(root: Path, log=print):
    cats = Path(root) / load_cfg(root)["paths"]["catalogs"]
    cats.mkdir(parents=True, exist_ok=True)
    if not any(cats.glob("*gz_hubble*")) and not any(cats.glob("*/*gz_hubble*")):
        log("[catalogues] downloading Galaxy Zoo: Hubble (about 25 MB) ...")
        _fetch(GZH_URL, cats / "gz_hubble_main.csv.gz")
    if not any(cats.glob("*candels*")) and not any(cats.glob("*/*candels*")) and not any(cats.glob("*k15*")) \
            and not any(cats.glob("*/*k15*")):
        log("[catalogues] downloading CANDELS Kartaltepe et al. 2015 (VizieR J/ApJS/221/11/table3) ...")
        _fetch(CANDELS_URL, cats / "candels_k15.csv")


def _fetch(url, out: Path):
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(out)


def count_near(root: Path, kind: str, targets, radius_deg: float) -> dict:
    """Catalogue rows within radius_deg of each target (GZH: labelled galaxies; CANDELS: expert labels)."""
    sys.path.insert(0, str(Path(root) / "src"))
    from .catalogs import PATTERNS, read_mapped, standardise
    cfg = load_cfg(root)
    cats = Path(root) / cfg["paths"]["catalogs"]
    path = next((h for pat in PATTERNS[kind] for h in sorted(cats.glob(pat)) + sorted(cats.glob(f"*/{pat}"))
                 if h.is_file()), None)
    if path is None:
        return {i: None for i in range(len(targets))}
    mapping = {k: v for k, v in cfg["catalogs"][kind]["columns"].items() if k in ("ra", "dec")}
    df = standardise(read_mapped(path, mapping), mapping, kind)
    out = {}
    for i, t in enumerate(targets):
        d = np.hypot((df.ra.to_numpy() - t.ra) * np.cos(np.radians(t.dec)), df.dec.to_numpy() - t.dec)
        out[i] = int((d < radius_deg).sum())
    return out


# ------------------------------------------------------------------ plan / prepare / run
def plan(root, targets, filters="auto", instrument="ACS/WFC", search_radius_deg=0.03, allow_single=True,
         mode="auto", min_gzh_labels=None, log=print) -> Plan:
    root = Path(root)
    cfg = load_cfg(root).get("field_runner", {})
    min_gzh = int(min_gzh_labels if min_gzh_labels is not None else cfg.get("min_gzh_labels", 150))
    p = Plan(targets=list(targets), instrument=instrument, filters=filters)
    for i, t in enumerate(p.targets):
        ch = archive.search(t.ra, t.dec, filters, instrument, search_radius_deg, allow_single, log=log)
        p.choices[i] = [asdict(c) for c in ch]
    try:
        ensure_catalogues(root, log)
    except Exception as e:                     # no internet for catalogues: decide from what is there
        log(f"[catalogues] could not download ({e!r}); using the files already in data/catalogs")
    fp = float(cfg.get("footprint_radius_deg", 0.028))
    p.gzh_counts = count_near(root, "gzh", p.targets, fp)
    p.candels_counts = count_near(root, "candels", p.targets, fp)
    n_gzh = sum(v or 0 for v in p.gzh_counts.values())
    if mode in ("train", "apply"):
        p.mode, p.reason = mode, "chosen by the user"
    elif n_gzh >= min_gzh:
        p.mode, p.reason = "train", f"{n_gzh} Galaxy Zoo: Hubble galaxies in the field(s) (>= {min_gzh})"
    else:
        p.mode, p.reason = "apply", (f"only {n_gzh} Galaxy Zoo: Hubble galaxies in the field(s) (< {min_gzh}): "
                                     "the frozen model bundle classifies the galaxies")
    log(f"[plan] mode = {p.mode}: {p.reason}")
    for i, t in enumerate(p.targets):
        log(f"[plan] {t.name or f'target {i + 1}'} ({t.ra}, {t.dec}): {len(p.choices[i])} band(s), "
            f"GZH {p.gzh_counts.get(i)}, CANDELS {p.candels_counts.get(i)}")
    return p


def newest_bundle(root: Path) -> Path | None:
    cards = sorted(Path(root).glob("models/*/model_card.json"), key=lambda p: p.stat().st_mtime)
    return cards[-1].parent if cards else None


def archive_previous_run(root: Path, log=print) -> Path | None:
    """Move data/raw/* and outputs/ to runs/<timestamp>/ (nothing is deleted)."""
    root = Path(root)
    cfg = load_cfg(root)
    raw, outs = root / cfg["paths"]["raw"], root / cfg["paths"]["outputs"]
    files = [f for f in raw.glob("*") if f.is_file() and f.name.lower() != "readme.md"] if raw.exists() else []
    has_out = outs.exists() and any(outs.iterdir())
    if not files and not has_out:
        return None
    dest = root / "runs" / time.strftime("%Y-%m-%d_%H%M%S")
    (dest / "raw").mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), dest / "raw" / f.name)
    if has_out:
        shutil.move(str(outs), dest / "outputs")
        outs.mkdir(parents=True, exist_ok=True)
    log(f"[archive] previous run moved to {dest.relative_to(root)} ({len(files)} image file(s){', outputs' if has_out else ''})")
    return dest


def prepare(root, p: Plan, fresh=True, model_bundle=None, download_images=True, log=print) -> str:
    """Download what the plan chose and write config_run.yaml. Returns the config file name to run with."""
    root = Path(root)
    cfg = load_cfg(root)
    if fresh:
        archive_previous_run(root, log)
    raw = root / cfg["paths"]["raw"]
    cats = root / cfg["paths"]["catalogs"]
    if download_images:
        for i, t in enumerate(p.targets):
            ch = [archive.Choice(**c) for c in p.choices.get(i, [])]
            if not ch:
                log(f"[prepare] no images for target {i + 1} ({t.ra}, {t.dec}) - skipped")
                continue
            archive.download(ch, raw, log)
            try:
                archive.gaia_cone(t.ra, t.dec, float(cfg.get("field_runner", {}).get("gaia_radius_deg", 0.06)), cats, log)
            except Exception as e:
                log(f"[gaia] cone failed ({e!r}); Stage 2c will ask Gaia online instead")
                cfg["catalogs"]["gaia"]["query_online"] = True
    cfg.setdefault("supervised", {})["mode"] = p.mode
    if p.mode == "apply":
        b = Path(model_bundle) if model_bundle else (Path(cfg["supervised"]["model_bundle"])
                                                      if cfg["supervised"].get("model_bundle") else newest_bundle(root))
        if b is None:
            raise RuntimeError("apply mode needs a model bundle (models/<name>/). Export one from a training run: "
                               "python tools/export_model.py --name <name>")
        cfg["supervised"]["model_bundle"] = str(b if b.is_absolute() else b)
        log(f"[prepare] apply mode with model bundle {cfg['supervised']['model_bundle']}")
    (root / "config_run.yaml").write_text(
        "# Written by p2p.field.prepare() for this run - edit config.yaml, not this file.\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (root / cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    (root / cfg["paths"]["outputs"] / "field_request.json").write_text(json.dumps(p.to_json(), indent=1))
    return "config_run.yaml"


def run_pipeline_stream(root, config_file="config_run.yaml", start=None, stop=None, keep_going=False):
    """Run run_pipeline.py in a child process and yield its output line by line."""
    cmd = [sys.executable, "-u", "run_pipeline.py", "--config", config_file]
    if start:
        cmd += ["--from", str(start)]
    if stop:
        cmd += ["--to", str(stop)]
    if keep_going:
        cmd.append("--keep-going")
    proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        yield line
    proc.wait()
    yield f"\n[run] finished with exit code {proc.returncode}\n"
