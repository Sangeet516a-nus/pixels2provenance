#!/usr/bin/env python
"""Explain one stage: its purpose (from the stage file), what it wrote, and the
headline numbers from its JSON outputs.

    python tools/show_stage.py 1c
    python tools/show_stage.py 6a --config my_other_config.yaml
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2p.context import STAGES, Ctx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    info = {k: (m, d, t) for k, m, d, t in STAGES}
    if a.stage not in info:
        sys.exit(f"unknown stage {a.stage}; choose from {list(info)}")
    mod, folder, title = info[a.stage]
    ctx = Ctx(ROOT, a.config)
    doc = importlib.import_module(f"p2p.stages.{mod}").__doc__
    print(f"Stage {a.stage} - {title}\nsrc/p2p/stages/{mod}.py\n")
    print(doc.strip(), "\n")
    out = ctx.out / folder
    if not out.exists():
        print(f"(no outputs yet in {ctx.rel(out)} - run: python run_pipeline.py --only {a.stage})")
        return
    print(f"Files in {ctx.rel(out)}:")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:45s} {f.stat().st_size / 1e6:8.2f} MB")
    for f in sorted(out.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        text = json.dumps(d, indent=1, default=str)
        print(f"\n--- {f.name} ---\n" + (text if len(text) < 2500 else text[:2500] + "\n ... (truncated)"))
    log = ctx.out / "run_log.json"
    if log.exists():
        r = json.loads(log.read_text()).get(a.stage)
        if r:
            print(f"\nlast run: {r['status']} in {r['seconds']} s at {r['finished']}")


if __name__ == "__main__":
    main()
