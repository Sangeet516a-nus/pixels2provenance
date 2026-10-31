"""Sealing of the EXTERNAL data (the one switch: evaluation.seal_external in config.yaml).

Train, dev, test and buffer galaxies go through every phase (0-9), Phase 6 (in-domain test) and Phase 7 (taxonomy
ceiling, cross-catalogue test, generalisation gap) included. The EXTERNAL galaxies (GOODS-N-7 and the external stripes of
GOODS-S, GOODS-N-2 and GOODS-N-4) are processed like any new field - detected, measured, classified, written to the
catalogue - but while seal_external is true no stage scores them against their labels, and Stage 3a does not show
their class mix. When the model is final set evaluation.seal_external: false and run
`python run_pipeline.py --from 7a`: Phase 7 then includes the external galaxies, and Stage 7b also scores their GZH
labels (same labels, new sky).

Older keys still work: evaluation.test_sealed (true/false) sets seal_external; evaluation.seal_test is ignored
(the test is always scored). Apply mode is never sealed (the bundle never saw this field).

Kept in its own module (not modes.py) so that a running pipeline picks it up at the first evaluation stage."""
from __future__ import annotations

from .modes import is_apply

SPLITS = ("train", "dev", "test", "buffer")


def external_sealed(ctx) -> bool:
    if is_apply(ctx):
        return False
    ev = ctx.cfg.get("evaluation", {}) or {}
    if "seal_external" in ev:
        return bool(ev["seal_external"])
    if "test_sealed" in ev:                    # older config
        return bool(ev["test_sealed"])
    return True


def test_sealed(ctx) -> bool:
    """The test galaxies are always scored (kept for older callers)."""
    return False


def scored_splits(ctx) -> list:
    """split_tag values whose labels may be scored: every split, external only when unsealed."""
    return list(SPLITS) + ([] if external_sealed(ctx) else ["external"])


def require_unsealed(ctx, stage: str, which: str = "external"):
    """Skip `stage` while the external data are sealed (only for stages that score external data alone)."""
    if which == "external" and external_sealed(ctx):
        from .context import StageSkipped
        raise StageSkipped(f"the external galaxies are SEALED (evaluation.seal_external: true) - stage {stage} "
                           "scores them only once the model is final: set evaluation.seal_external: false, then "
                           "python run_pipeline.py --from 7a")
