"""Stage 4c - Probability calibration (temperature scaling).

Purpose : make predicted confidence match empirical accuracy.
Method  : one temperature T for the soft-voted morphology log-probabilities
          (T < 1 sharpens - an averaged ensemble is usually under-confident) and
          Platt scaling p = sigmoid(a z + b) for the merger logit (the positive-class
          weight used in training shifts the logits; a temperature cannot remove a
          shift). Both are fitted on the DEV split by minimising the NLL. Reported with the Expected
          Calibration Error (ECE) before and after.
Output  : temperature.json, reliability_dev.png
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..context import Ctx, save_json
from ..features import TabularPrep, load_cutouts, make_tensors, supervised_splits
from ..inference import ensemble_predict, load_models
from ..nets import logit, softmax_np
from ..utils import expected_calibration_error


def fit_platt(z, y):
    """p = sigmoid(a z + b): a re-scales AND shifts the logit. The merger head is
    trained with a positive-class weight, which shifts every logit up; a single
    temperature cannot undo a shift, Platt scaling can."""
    from scipy.optimize import minimize
    y = np.asarray(y, float)

    def nll(ab):
        p = 1 / (1 + np.exp(-(ab[0] * z + ab[1])))
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def fit_temperature(logits, y):
    y = np.asarray(y, int)
    nll = lambda lt: -np.mean(np.log(softmax_np(logits / np.exp(lt))[np.arange(len(y)), y] + 1e-12))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def run(ctx: Ctx):
    out = ctx.stage_dir("4c")
    splits = supervised_splits(ctx)
    prep = TabularPrep.load(ctx.require("4a", "tab_prep.joblib"))
    dev = make_tensors(splits["dev"], load_cutouts(ctx), prep)
    ens = ensemble_predict(ctx, dev, load_models(ctx))
    y, ym = dev["y"].numpy(), dev["ym"].numpy().astype(int)
    L = np.log(np.clip(ens["morph"], 1e-12, 1))
    T_m = fit_temperature(L, y)
    zm = logit(ens["merger"])
    a, b = fit_platt(zm, ym) if 0 < ym.sum() < len(ym) else (1.0, 0.0)
    Pc = softmax_np(L / T_m)
    two = lambda p: np.column_stack([1 - p, p])
    pm_c = 1 / (1 + np.exp(-(a * zm + b)))
    rep = {"T_morph": round(T_m, 4), "merger_platt_a": round(a, 4), "merger_platt_b": round(b, 4),
           "dev_ece_morph_before": round(expected_calibration_error(ens["morph"], y), 4),
           "dev_ece_morph_after": round(expected_calibration_error(Pc, y), 4),
           "dev_ece_merger_before": round(expected_calibration_error(two(ens["merger"]), ym), 4),
           "dev_ece_merger_after": round(expected_calibration_error(two(pm_c), ym), 4), "n_dev": int(len(y)),
           "dev_merger_rate_true": round(float(ym.mean()), 4),
           "dev_merger_rate_predicted_before": round(float((ens["merger"] >= 0.5).mean()), 4),
           "dev_merger_rate_predicted_after": round(float((pm_c >= 0.5).mean()), 4)}
    # decision threshold for the binary merger FLAG: flag the same fraction of dev galaxies as are
    # true mergers (prevalence matching). The probability p_merger itself stays calibrated.
    if 0 < ym.sum() < len(ym):
        thr = float(np.quantile(pm_c, 1 - ym.mean()))
        flag = pm_c >= thr
        rep["merger_threshold"] = round(thr, 4)
        rep["dev_merger_precision_at_threshold"] = round(float((flag & (ym == 1)).sum() / max(flag.sum(), 1)), 4)
        rep["dev_merger_recall_at_threshold"] = round(float((flag & (ym == 1)).sum() / ym.sum()), 4)
    else:
        rep["merger_threshold"] = 0.5
    save_json(rep, out / "temperature.json")
    _plot(ens["morph"], Pc, y, out)
    ctx.log.info(f"[Stage 4c] {rep}")
    return rep


def _plot(P0, P1, y, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for P, lab in ((P0, "before"), (P1, "after")):
        conf, acc = P.max(1), (P.argmax(1) == y)
        idx = np.digitize(conf, np.linspace(0, 1, 11)) - 1
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [acc[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, "o-", label=f"{lab} T-scaling")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Stage 4c: dev reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability_dev.png", dpi=110)
    plt.close(fig)
