"""Stage 7b - Cross-catalogue evaluation (run once).

Purpose : how well does the frozen model generalise to an independent expert scheme?
Method  : the frozen, calibrated ensemble predicts every external galaxy with a
          CANDELS label; galaxies in the GZH overlap (used for the ceiling) and
          expert point sources are excluded. Predictions are mapped to the
          shared taxonomy and scored with Macro-F1 (+ 95% bootstrap interval).
Output  : crosscat_predictions.csv, cross_catalogue_metrics.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from ..context import Ctx, StageSkipped, save_json
from ..features import TabularPrep, galaxy_table, load_cutouts, make_tensors
from ..inference import calibrated_predict, load_models
from ..nets import MORPH_CLASSES
from ..utils import bootstrap_ci
from .stage7a_taxonomy import COMMON, GZH_TO_COMMON


def run(ctx: Ctx):
    out = ctx.stage_dir("7b")
    p = ctx.path("7a", "candels_objects.csv")
    if not p.exists():
        raise StageSkipped("Stage 7a produced no CANDELS objects.")
    c = pd.read_csv(p)
    c["in_gzh_overlap"] = c.in_gzh_overlap.astype(str).str.lower().eq("true")
    test = c[~c.in_gzh_overlap & c.k15_common.notna()]
    if test.empty:
        raise StageSkipped("No CANDELS-labelled external galaxies outside the GZH overlap.")
    g = galaxy_table(ctx).set_index("object_id").loc[test.object_id].reset_index()
    data = make_tensors(g, load_cutouts(ctx), TabularPrep.load(ctx.require("4a", "tab_prep.joblib")), with_labels=False)
    res = calibrated_predict(ctx, data, load_models(ctx))
    pred = np.array(MORPH_CLASSES)[res["morph"].argmax(1)]
    test = test.assign(pred_class=pred, pred_common=pd.Series(pred).map(GZH_TO_COMMON).to_numpy(),
                       p_merger=res["merger"])
    test.to_csv(out / "crosscat_predictions.csv", index=False)
    labs = {c_: i for i, c_ in enumerate(COMMON)}
    yt = test.k15_common.map(labs).to_numpy()
    yp = test.pred_common.map(labs).to_numpy()
    f1 = lambda a, b: f1_score(a, b, labels=sorted(set(np.asarray(a).tolist()) | set(np.asarray(b).tolist())),
                               average="macro", zero_division=0)
    m = {"xcat_macro_f1": round(float(f1(yt, yp)), 4),
         "xcat_macro_f1_ci95": bootstrap_ci(lambda a, P: f1(a, P), yt, yp, seed=ctx.seed),
         "n_test": int(len(test)), "n_excluded_overlap": int(c.in_gzh_overlap.sum()),
         "n_excluded_point_source": int(c.k15_common.isna().sum()),
         "per_class": classification_report(test.k15_common, test.pred_common, labels=list(COMMON),
                                            output_dict=True, zero_division=0)}
    save_json(m, out / "cross_catalogue_metrics.json")
    ctx.log.info(f"[Stage 7b] cross-catalogue Macro-F1 = {m['xcat_macro_f1']} on n = {m['n_test']}\n" +
                 pd.crosstab(test.k15_common, test.pred_common, rownames=["CANDELS"], colnames=["model"]).to_string())
    return {"xcat_macro_f1": m["xcat_macro_f1"]}
