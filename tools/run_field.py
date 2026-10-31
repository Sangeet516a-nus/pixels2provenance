#!/usr/bin/env python
"""Give sky positions; the images are found, downloaded and the whole pipeline runs (needs internet).

    python tools/run_field.py --target 53.12,-27.80 --search-only            # what would be downloaded
    python tools/run_field.py --target 53.12,-27.80                          # download + run everything
    python tools/run_field.py --target 150.12,2.21 --target 189.23,62.24     # several fields together
    python tools/run_field.py --target 34.4,-5.2 --mode apply --model-bundle models/goods_s_2026-10-24

Mode: 'auto' trains on the field when it has enough Galaxy Zoo: Hubble labels (field_runner.min_gzh_labels),
otherwise applies a frozen model bundle (tools/export_model.py). The previous run (data/raw files and outputs/)
is moved to runs/<timestamp>/ first unless --keep-previous. The run uses config_run.yaml (config.yaml +
the choices made here).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_target(s: str):
    from p2p.field import parse_targets
    t = parse_targets(s)
    if len(t) != 1:
        raise argparse.ArgumentTypeError("use RA,Dec[,name], e.g. 53.12,-27.80 or 03:32:28.8,-27:48:00,GOODS-S")
    return t[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", type=parse_target, required=True, help="RA,Dec[,name] in degrees")
    ap.add_argument("--instrument", default="ACS/WFC")
    ap.add_argument("--filters", nargs="+", default=["auto"])
    ap.add_argument("--radius", type=float, default=0.03, help="MAST search radius [deg]")
    ap.add_argument("--combined-only", action="store_true", help="never download single-exposure files")
    ap.add_argument("--mode", choices=["auto", "train", "apply"], default="auto")
    ap.add_argument("--model-bundle", help="models/<name> for apply mode (default: config, else the newest)")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--keep-previous", action="store_true", help="do not move the previous run to runs/")
    ap.add_argument("--no-run", action="store_true", help="download only")
    ap.add_argument("--from", dest="start", help="first stage to run (default: 0)")
    a = ap.parse_args()
    from p2p import field
    filters = "auto" if a.filters == ["auto"] else [f.upper() for f in a.filters]
    p = field.plan(ROOT, a.target, filters, a.instrument, a.radius, allow_single=not a.combined_only, mode=a.mode)
    if a.search_only:
        return
    if not any(p.choices.values()):
        sys.exit("no HST images cover the target(s)")
    cfg_file = field.prepare(ROOT, p, fresh=not a.keep_previous, model_bundle=a.model_bundle)
    if a.no_run:
        print(f"downloaded; run with: python run_pipeline.py --config {cfg_file}")
        return
    code = 0
    for line in field.run_pipeline_stream(ROOT, cfg_file, start=a.start):
        print(line, end="", flush=True)
        if line.startswith("[run] finished with exit code"):
            code = int(line.strip().split()[-1])
    sys.exit(code)


if __name__ == "__main__":
    main()
