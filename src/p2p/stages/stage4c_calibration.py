"""Stage 4c - Probability calibration of the Stage 4b XGBoost models (no network trained here).

Model   : 3 parameters. Morphology: temperature scaling p = softmax(log P / T) - one T for all 5 classes
          (T < 1 sharpens an under-confident average, T > 1 softens; the predicted class, hence the accuracy,
          does not change). Merger: Platt scaling p = sigmoid(a z + b) on the merger logit z (the positive-class
          weight used in training shifts every logit; a temperature cannot undo a shift).
Fit     : minimum negative log-likelihood on the OUT-OF-FOLD predictions of the pooled train + dev galaxies
          (Stage 4b; each galaxy predicted by a model that never saw it).
Metrics : accuracy, log-loss and Expected Calibration Error (ECE) before / after; merger flag threshold by
          prevalence matching. A negative Platt slope (the head ranks mergers below non-mergers) -> base rate.
Output  : temperature.json, reliability_dev.png
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..context import Ctx, save_json
from ..nets import logit, nll, softmax_np
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
    from ..modes import install_bundle, is_apply
    if is_apply(ctx):                       # no training: copy the frozen model bundle
        return install_bundle(ctx, "4c")
    out = ctx.stage_dir("4c")
    d = np.load(ctx.require("4b", "oof_fusion.npz"))           # out-of-fold predictions, pooled train + dev
    ens = {"morph": d["morph"], "merger": d["merger"]}
    y, ym = d["y"].astype(int), d["ym"].astype(int)
    source = "out-of-fold predictions, pooled train + dev"
    L = np.log(np.clip(ens["morph"], 1e-12, 1))
    T_m = fit_temperature(L, y)
    zm = logit(ens["merger"])
    merger_note = None
    if not ctx.path("4b", "xgb_merger.joblib").exists():        # no merger model: the base rate is the prediction
        a, b = 0.0, float(logit(np.array([max(ym.mean(), 1e-6)]))[0])
        merger_note = f"no merger model (too few mergers): p_merger = base rate {ym.mean():.3f}"
    elif 0 < ym.sum() < len(ym):
        a, b = fit_platt(zm, ym)
    else:
        a, b = 1.0, 0.0
    if merger_note is None and a <= 0 and 0 < ym.sum() < len(ym):
        # a negative slope would INVERT the merger score: the head has no usable ranking signal on these
        # objects (typically a handful of mergers). Report the base rate instead of an inverted score.
        merger_note = (f"Platt slope {a:.3f} <= 0: the merger head does not rank mergers above non-mergers on the "
                       f"calibration set ({int(ym.sum())} mergers) - p_merger set to the base rate {ym.mean():.3f}")
        a, b = 0.0, float(logit(np.array([ym.mean()]))[0])
        ctx.log.info(f"[Stage 4c] WARNING: {merger_note}")
    Pc = softmax_np(L / T_m)
    two = lambda p: np.column_stack([1 - p, p])
    pm_c = 1 / (1 + np.exp(-(a * zm + b)))
    rep = {"T_morph": round(T_m, 4), "merger_platt_a": round(a, 4), "merger_platt_b": round(b, 4),
           "accuracy": round(float((Pc.argmax(1) == y).mean()), 4),
           "log_loss_before": round(nll(y, ens["morph"]), 4), "log_loss_after": round(nll(y, Pc), 4),
           "dev_ece_morph_before": round(expected_calibration_error(ens["morph"], y), 4),
           "dev_ece_morph_after": round(expected_calibration_error(Pc, y), 4),
           "dev_ece_merger_before": round(expected_calibration_error(two(ens["merger"]), ym), 4),
           "dev_ece_merger_after": round(expected_calibration_error(two(pm_c), ym), 4), "n_dev": int(len(y)),
           "dev_merger_rate_true": round(float(ym.mean()), 4),
           "dev_merger_rate_predicted_before": round(float((ens["merger"] >= 0.5).mean()), 4),
           "dev_merger_rate_predicted_after": round(float((pm_c >= 0.5).mean()), 4),
           "fitted_on": source, "merger_note": merger_note}
    # decision threshold for the binary merger FLAG: flag the same fraction of dev galaxies as are
    # true mergers (prevalence matching). The probability p_merger itself stays calibrated.
    if merger_note:
        rep["merger_threshold"] = 1.0               # no usable signal: raise no merger flags
    elif 0 < ym.sum() < len(ym):
        thr = float(np.quantile(pm_c, 1 - ym.mean()))
        flag = pm_c >= thr
        rep["merger_threshold"] = round(thr, 4)
        rep["dev_merger_precision_at_threshold"] = round(float((flag & (ym == 1)).sum() / max(flag.sum(), 1)), 4)
        rep["dev_merger_recall_at_threshold"] = round(float((flag & (ym == 1)).sum() / ym.sum()), 4)
    else:
        rep["merger_threshold"] = 0.5
    save_json(rep, out / "temperature.json")
    _plot(ens["morph"], Pc, y, out)
    ctx.log.info(f"[Stage 4c] calibration model: T = {rep['T_morph']} (morphology), Platt a = {rep['merger_platt_a']}, "
                 f"b = {rep['merger_platt_b']} (merger) - fitted on {len(y)} out-of-fold galaxies")
    ctx.log.info(f"[Stage 4c] accuracy {rep['accuracy']} (unchanged by T) | log-loss {rep['log_loss_before']} -> "
                 f"{rep['log_loss_after']} | ECE morphology {rep['dev_ece_morph_before']} -> {rep['dev_ece_morph_after']}"
                 f" | ECE merger {rep['dev_ece_merger_before']} -> {rep['dev_ece_merger_after']} | merger threshold "
                 f"{rep['merger_threshold']}")
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
    ax.set_title("Stage 4c: reliability (calibration set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability_dev.png", dpi=110)
    plt.close(fig)
