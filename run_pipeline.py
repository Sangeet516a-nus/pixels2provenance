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

# pandas >= 2 prints "DataFrame is highly fragmented" when many catalogue columns are added one by one
# (Stage 8 builds 91). It is a speed hint only - the results are identical - so it is not shown.
try:
    from pandas.errors import PerformanceWarning
    warnings.simplefilter("ignore", PerformanceWarning)
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single stage, e.g. 1c")
    ap.add_argument("--from", dest="start", help="first stage to run")
    ap.add_argument("--to", dest="stop", help="last stage to run")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    ap.add_argument("--summary", action="store_true",
                    help="print a model.summary() of every CNN (built from the config, untrained) and exit")
    args = ap.parse_args()
    if args.summary:
        _model_summaries(Ctx(ROOT, args.config))
        return

    keys = [k for k, *_ in STAGES]
    for name, v in (("--from", args.start), ("--to", args.stop), ("--only", args.only)):
        if v is not None and v not in keys:
            sys.exit(f"Unknown stage {v!r} for {name}. Stages: {', '.join(keys)} (python run_pipeline.py --list)")
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
            _limit_foreign_openmp()
            summary = mod.run(ctx) or {}
        except StageSkipped as e:
            status, summary = "skipped", {"reason": str(e)}
            ctx.log.info(f"[Stage {k}] SKIPPED: {e}")
            _clear_stale(ctx, k, str(e))
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


def _model_summaries(ctx):
    """model.summary() of every network of the pipeline: the two autoencoders trained in Stages 1b / 2a and the two
    frozen ImageNet feature extractors of Stage 4a (built without ImageNet weights here: only the shapes matter)."""
    import torch
    from p2p.nets import ConvAutoencoder, EfficientNetB0, RepAutoencoder, ResNet18, model_summary
    qc, rep, sup = ctx.cfg["frame_qc"], ctx.cfg["representation"], ctx.cfg["supervised"]
    size = int(ctx.cfg.get("detection", {}).get("cutout_size", 128))
    bands = int(ctx.cfg["acquisition"].get("max_bands", 3))
    masks = bool(rep.get("band_mask_channels", True))
    tile, s4 = int(qc.get("tile", 128)), int(sup.get("input_size", 224))
    if int(qc.get("qc_version", 2)) == 1:
        m1 = ConvAutoencoder(in_ch=1, latent_dim=int(qc["latent_dim"]), size=tile)
    else:
        m1 = RepAutoencoder(bands=1, latent_dim=int(qc["latent_dim"]), size=tile,
                            channels=tuple(qc.get("channels", [16, 32, 64, 64])), mask_channels=False)
    m2 = RepAutoencoder(bands=bands, latent_dim=int(rep["latent_dim"]), size=size,
                        channels=tuple(rep.get("channels", [16, 32, 64, 64])), mask_channels=masks)
    x4 = torch.zeros(1, 3, s4, s4)
    models = [
        ("Stage 1b - frame-QC autoencoder", m1, (torch.zeros(1, 1, tile, tile),)),
        ("Stage 2a - representation autoencoder", m2, (torch.zeros(1, bands * (2 if masks else 1), size, size),)),
        (f"Stage 4a - ResNet-18 feature extractor (ImageNet, frozen; input {s4} px)", ResNet18(pretrained=False), (x4,)),
        (f"Stage 4a - EfficientNet-B0 feature extractor (ImageNet, frozen; input {s4} px)",
         EfficientNetB0(pretrained=False), (x4,)),
    ]
    txt = "\n\n".join(model_summary(m, *x, name=n) for n, m, x in models)
    txt += ("\n\nStage 4b - XGBoost (feature fusion): not a network; its summary (trees, leaves, parameters, "
            "importance by ENSEMBLE block) is written by Stage 4b to outputs/stage4b_xgboost_fusion/model_summary.txt")
    print(txt)
    path = ctx.out / "model_summaries.txt"
    path.write_text(txt + "\n")
    print(f"\nWritten to {ctx.rel(path)}")


def _limit_foreign_openmp():
    """On macOS, PyTorch ships its own libomp and scikit-learn uses the environment's (e.g. Anaconda's).
    Once Stage 2a has run torch, a multi-threaded OpenMP region in the second runtime (KMeans inside
    GaussianMixture, HDBSCAN, ...) crashes the process with a segmentation fault. Every OpenMP runtime
    that is NOT torch's is pinned to one thread; torch keeps its threads. Called after each stage
    import, because a runtime can only be limited once it is loaded."""
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        return
    for lib in ThreadpoolController().lib_controllers:
        if lib.user_api == "openmp" and "torch" not in Path(lib.filepath).parts:
            lib.set_num_threads(1)


KEEP_ON_SKIP = {"test_split_ledger.json"}


def _clear_stale(ctx, k, reason):
    """A skipped stage must not leave files from an EARLIER run behind: later stages would read them."""
    from p2p.context import STAGE_DIR
    d = ctx.out / STAGE_DIR[k]
    if d.exists():
        import shutil
        for f in d.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            elif f.name not in KEEP_ON_SKIP:
                f.unlink()
        (d / "SKIPPED.txt").write_text(reason + "\n")


def _save(runlog, k, status, t0, summary, path):
    runlog[k] = {"status": status, "seconds": round(time.time() - t0, 1),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary}
    save_json(runlog, path)


if __name__ == "__main__":
    main()
