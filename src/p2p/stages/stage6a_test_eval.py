"""Stage 6a - In-domain test (run once).

Purpose : the final, honest in-domain score.
Method  : the frozen, calibrated model (Stage 4b fusion XGBoost + Stage 4c calibration) is run on the untouched GZH test
          split: Macro-F1, balanced accuracy, per-class precision/recall,
          one-vs-rest PR-AUC, merger PR-AUC and ECE, each with a 95% bootstrap
          interval (small test sets need error bars), scored against the tiered
          success criteria. Every run is appended to a ledger, so the report can
          state how many times the test split was looked at.
Output  : test_metrics.json, test_predictions.csv, confusion_matrix_test.png, test_split_ledger.json
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score

from ..context import Ctx, load_json, save_json
from ..features import supervised_splits
from ..inference import predict_galaxies
from ..nets import MORPH_CLASSES, N_CLASSES, morph_metrics
from ..utils import bootstrap_ci, expected_calibration_error


def tier(v, minimum, target, stretch, lower_better=False):
    if v is None or not np.isfinite(v):
        return "n/a"
    if lower_better:
        v, minimum, target, stretch = -v, -minimum, -target, -stretch
    return "Stretch" if v >= stretch else "Target" if v >= target else "Minimum" if v > minimum else "Below minimum"


def _ci(c):
    return f"{c[0]:.3f}-{c[1]:.3f}" if c and np.isfinite(c[0]) else "n/a"


def run(ctx: Ctx):
    from ..sealing import require_unsealed
    require_unsealed(ctx, "6a")
    out = ctx.stage_dir("6a")
    ledger_p = out / "test_split_ledger.json"
    ledger = load_json(ledger_p) if ledger_p.exists() else {"evaluations": []}
    if ledger["evaluations"]:
        ctx.log.info(f"[Stage 6a] WARNING: the test split was already evaluated {len(ledger['evaluations'])} time(s). "
                     "Report the FIRST result and do not tune on these numbers.")
    splits = supervised_splits(ctx)
    tdf = splits["test"]
    res = predict_galaxies(ctx, tdf)
    y, ym, P = tdf.gz_label_id.astype(int).to_numpy(), tdf.gz_merger_flag.fillna(0).astype(int).to_numpy(), res["morph"]
    pred = P.argmax(1)
    from ..modes import card, is_apply
    if is_apply(ctx) and card(ctx).get("train_class_counts"):
        cc = card(ctx)["train_class_counts"]                      # the bundle's own training set
        maj = int(np.argmax([cc.get(c, 0) for c in MORPH_CLASSES]))
    else:
        maj = np.bincount(splits["train"].gz_label_id, minlength=N_CLASSES).argmax()
    maj_f1 = morph_metrics(y, np.eye(N_CLASSES)[np.full(len(y), maj)])["macro_f1"]
    m = morph_metrics(y, P, ym, res["merger"])
    f1 = lambda yy, PP: f1_score(yy, PP.argmax(1), average="macro",
                                 labels=sorted(set(yy.tolist()) | set(PP.argmax(1).tolist())), zero_division=0)
    m["macro_f1_ci95"] = bootstrap_ci(f1, y, P, seed=ctx.seed)
    present = np.eye(N_CLASSES)[y].sum(0) > 0
    m["morph_pr_auc_ovr"] = round(float(average_precision_score(np.eye(N_CLASSES)[y][:, present], P[:, present],
                                                                 average="macro")), 4)
    m["ece"] = round(expected_calibration_error(P, y), 4)
    m["ece_ci95"] = bootstrap_ci(lambda yy, PP: expected_calibration_error(PP, yy), y, P, seed=ctx.seed)
    if m.get("merger_pr_auc") is not None and 0 < ym.sum() < len(ym):
        ap = lambda a, p: average_precision_score(a, p) if 0 < a.sum() < len(a) else float("nan")
        lo, hi = bootstrap_ci(ap, ym, np.asarray(res["merger"], float), seed=ctx.seed)
        m["merger_pr_auc_ci95"] = (lo, hi)
        m["n_merger_test"] = int(ym.sum())
    m["majority_class_macro_f1"] = maj_f1
    m["per_class"] = classification_report(y, pred, labels=list(range(N_CLASSES)), target_names=list(MORPH_CLASSES),
                                           output_dict=True, zero_division=0)
    m["tiers"] = {"Morphology Macro-F1": tier(m["macro_f1"], maj_f1, 0.65, 0.75),
                  "Merger PR-AUC": tier(m.get("merger_pr_auc"), m.get("merger_prevalence", 0), 0.50, 0.65),
                  "Calibration ECE": tier(m["ece"], 0.10, 0.05, 0.03, lower_better=True)}
    save_json(m, out / "test_metrics.json")
    ledger["evaluations"].append({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                  "macro_f1": m["macro_f1"], "n_test": int(len(y))})
    save_json(ledger, ledger_p)
    pr = pd.DataFrame({"object_id": tdf.object_id.to_numpy(), "y_true": y, "y_pred": pred, "p_merger": res["merger"]})
    for k, c in enumerate(MORPH_CLASSES):
        pr[f"p_{c}"] = P[:, k]
    pr.to_csv(out / "test_predictions.csv", index=False)
    _confusion(y, pred, out)
    ctx.log.info(f"[Stage 6a] test n={len(y)}: accuracy {m['accuracy']} | Macro-F1 {m['macro_f1']} (95% CI {m['macro_f1_ci95'][0]:.3f}-"
                 f"{m['macro_f1_ci95'][1]:.3f}) | balanced acc {m['balanced_acc']} | merger PR-AUC "
                 f"{m.get('merger_pr_auc')} (95% CI {_ci(m.get('merger_pr_auc_ci95'))}, "
                 f"{m.get('n_merger_test', 0)} mergers) | ECE {m['ece']} (95% CI {_ci(m['ece_ci95'])}) | "
                 f"majority-class F1 {maj_f1}")
    ctx.log.info(f"[Stage 6a] tiers: {m['tiers']}")
    return {"macro_f1": m["macro_f1"], "tiers": m["tiers"]}


def _confusion(y, pred, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(MORPH_CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(MORPH_CLASSES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("GZH label")
    ax.set_title("Stage 6a: in-domain test (row-normalised, counts)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix_test.png", dpi=110)
    plt.close(fig)
