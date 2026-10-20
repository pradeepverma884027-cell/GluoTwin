"""Shared code for all train_*.py scripts.
Input = your ORIGINAL csv files (GlucoTwin.csv, smart_wearable.csv, t1dm.csv, t2dm.csv, ohhim.csv) in --raw_dir.
First run builds the model-ready data automatically (prep_pipeline.py); later runs reuse it from --processed_dir."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, roc_auc_score)

# `current` = column with glucose right now (for the "same as now" baseline)
DATASETS = {"smart_wearable": dict(current=None), "glucotwin": dict(current="Dexcom_t0"),
            "cgm_t1dm_t2dm": dict(current="cgm"), "ohhim": dict(current="glucose")}
SEQ = {  # oldest -> newest glucose columns, used by the GRU
    "cgm_t1dm_t2dm": ["cgm_lag_120m", "cgm_lag_90m", "cgm_lag_60m", "cgm_lag_45m", "cgm_lag_30m", "cgm_lag_15m", "cgm"],
    "ohhim": ["glucose_lag_60m", "glucose_lag_45m", "glucose_lag_30m", "glucose_lag_15m", "glucose_lag_10m",
              "glucose_lag_5m", "glucose"]}


def add_args(ap, datasets=None):
    ap.add_argument("--raw_dir", default="data", help="folder with your original csv files")
    ap.add_argument("--processed_dir", default="processed", help="where model-ready data is stored / reused")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--rebuild", action="store_true", help="redo data preparation even if processed files exist")
    ap.add_argument("--datasets", nargs="*", default=datasets or list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--targets", nargs="*", help="only these target columns (default: all targets of the dataset)")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def prepare(a):
    proc = Path(a.processed_dir)
    need = [d for d in a.datasets if a.rebuild or not ((proc / f"{d}_model_ready.csv").exists()
                                                       and (proc / f"schema_{d}.json").exists())]
    if need:
        print(f"preparing data from {a.raw_dir}: {need}", flush=True)
        import prep_pipeline
        prep_pipeline.run_all(a.raw_dir, a.processed_dir, only=need)


def load(a, name):
    df = pd.read_csv(Path(a.processed_dir) / f"{name}_model_ready.csv")
    sc = json.load(open(Path(a.processed_dir) / f"schema_{name}.json"))
    return df, sc["features"], sc["targets"], sc


def task_of(df, t):
    return "clf" if df.loc[df["split"] == "train", t].nunique() <= 2 else "reg"


def splits(df):
    return {s: df[df["split"] == s] for s in ("train", "val", "test")}


def unscale(sc, col, x):
    m, s = sc["scaled_cols_mean_std"][col]
    return x * s + m


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[np.argmax(f1[:-1])]) if len(th) else 0.5


def evaluate(task, y, pred, thr=None):
    y = np.asarray(y)
    if task == "reg":
        return dict(RMSE=float(np.sqrt(mean_squared_error(y, pred))), MAE=float(mean_absolute_error(y, pred)))
    ok = len(set(y)) > 1
    out = dict(AUC=float(roc_auc_score(y, pred)) if ok else np.nan,
               PR_AUC=float(average_precision_score(y, pred)) if ok else np.nan)
    if thr is not None:
        yhat = (np.asarray(pred) >= thr).astype(int); tp = int(((yhat == 1) & (y == 1)).sum())
        out.update(precision=tp / max(1, int(yhat.sum())), recall=tp / max(1, int((y == 1).sum())), threshold=thr)
    return out


def default_fit(m, Xtr, ytr, Xva, yva):
    m.fit(Xtr, ytr)


def run_tabular(a, model_name, make, fit=None, with_same_as_now=False):
    """Train one sklearn-style model per (dataset, target); write metrics + saved model.
    make(task, pos_weight, args) -> estimator ; fit(model, Xtr, ytr, Xva, yva) optional (early stopping etc.)."""
    prepare(a)
    res = Path(a.results_dir); (res / "models").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.datasets:
        df, feats, targets, sc = load(a, name)
        S = splits(df)
        for t in (a.targets or targets):
            if t not in targets: continue
            task = task_of(df, t)
            pos = float(S["train"][t].mean()) if task == "clf" else None
            m = make(task, (1 - pos) / max(pos, 1e-6) if pos is not None else None, a)
            (fit or default_fit)(m, S["train"][feats], S["train"][t], S["val"][feats], S["val"][t])
            pv = (lambda d: m.predict(d[feats])) if task == "reg" else (lambda d: m.predict_proba(d[feats])[:, 1])
            thr = best_f1_threshold(S["val"][t], pv(S["val"])) if task == "clf" else None
            for s in ("val", "test"):
                rows.append(dict(dataset=name, model=model_name, target=t, split=s, **evaluate(task, S[s][t], pv(S[s]), thr)))
            cur = DATASETS[name]["current"]
            if with_same_as_now and cur and task == "reg" and ("glucose" in t.lower() or "cgm" in t.lower()):
                for s in ("val", "test"):
                    rows.append(dict(dataset=name, model="same-as-now", target=t, split=s,
                                     **evaluate("reg", S[s][t], unscale(sc, cur, S[s][cur].to_numpy()))))
            joblib.dump(m, res / "models" / f"{model_name}__{name}__{t}.joblib")
            print(f"trained {model_name} | {name} | {t}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(res / f"{model_name}_metrics.csv", index=False)
    print(out[out.split == "test"].round(3).to_string(index=False))
    return out
