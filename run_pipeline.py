#!/usr/bin/env python
"""Run the From Pixels to Provenance pipeline.

Examples (from the project folder):
    python run_pipeline.py                  # every stage, in order
    python run_pipeline.py --list           # show the stages
    python run_pipeline.py --only 1c        # one stage (its inputs must exist)
    python run_pipeline.py --from 2a        # from 2a to the end
    python run_pipeline.py --from 1a --to 1e
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from p2p.context import STAGES, Ctx, MissingInput, StageSkipped, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single stage, e.g. 1c")
    ap.add_argument("--from", dest="start", help="first stage to run")
    ap.add_argument("--to", dest="stop", help="last stage to run")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    args = ap.parse_args()

    keys = [k for k, *_ in STAGES]
    if args.list:
        for k, mod, folder, title in STAGES:
            print(f"  {k:>3}  {title:45s} -> outputs/{folder}/   (src/p2p/stages/{mod}.py)")
        return
    if args.only:
        todo = [args.only]
    else:
        i0 = keys.index(args.start) if args.start else 0
        i1 = keys.index(args.stop) if args.stop else len(keys) - 1
        todo = keys[i0:i1 + 1]
    unknown = [k for k in todo if k not in keys]
    if unknown:
        sys.exit(f"Unknown stage(s): {unknown}. Use --list.")

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ctx = Ctx(ROOT, args.config)
    runlog_path = ctx.out / "run_log.json"
    runlog = json.loads(runlog_path.read_text()) if runlog_path.exists() else {}
    info = {k: (m, t) for k, m, _, t in STAGES}

    for k in todo:
        mod_name, title = info[k]
        ctx.log.info("=" * 78)
        ctx.log.info(f"Stage {k} - {title}")
        ctx.log.info("=" * 78)
        t0 = time.time()
        status = "ok"
        try:
            mod = importlib.import_module(f"p2p.stages.{mod_name}")
            summary = mod.run(ctx) or {}
        except StageSkipped as e:
            status, summary = "skipped", {"reason": str(e)}
            ctx.log.info(f"[Stage {k}] SKIPPED: {e}")
        except MissingInput as e:
            status, summary = "missing-input", {"reason": str(e)}
            ctx.log.info(f"[Stage {k}] CANNOT RUN: {e}")
            if not args.keep_going:
                _save(runlog, k, status, t0, summary, runlog_path)
                sys.exit(1)
        except Exception as e:
            status, summary = "failed", {"error": repr(e)}
            ctx.log.info(traceback.format_exc())
            if not args.keep_going:
                _save(runlog, k, status, t0, summary, runlog_path)
                sys.exit(1)
        _save(runlog, k, status, t0, summary, runlog_path)
        ctx.log.info(f"[Stage {k}] {status} in {time.time() - t0:.1f} s")
    ctx.log.info(f"Done. Outputs in {ctx.rel(ctx.out)} (final catalogue: stage8_catalogue/catalog.csv)")


def _save(runlog, k, status, t0, summary, path):
    runlog[k] = {"status": status, "seconds": round(time.time() - t0, 1),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary}
    save_json(runlog, path)


if __name__ == "__main__":
    main()
