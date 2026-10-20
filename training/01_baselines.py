"""Baselines you must beat: 'same as now' rule + Ridge / Logistic Regression.
Usage: python 01_baselines.py [dataset ...]"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from common import DATASETS, RESULTS, best_f1_threshold, evaluate, load, splits, task_of, unscale

rows = []
for name in (sys.argv[1:] or DATASETS):
    df, feats, targets, sc = load(name)
    S = splits(df)
    for t in targets:
        task = task_of(df, t)
        if task == "reg":
            m = Ridge(alpha=1.0).fit(S["train"][feats], S["train"][t]); pv = lambda d: m.predict(d[feats])
        else:
            m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(S["train"][feats], S["train"][t])
            pv = lambda d: m.predict_proba(d[feats])[:, 1]
        thr = best_f1_threshold(S["val"][t], pv(S["val"])) if task == "clf" else None
        for s in ("val", "test"):
            rows.append(dict(dataset=name, model="Ridge/Logistic", target=t, split=s,
                             **evaluate(task, S[s][t], pv(S[s]), thr)))
        cur = DATASETS[name]["current"]
        if cur and task == "reg" and ("glucose" in t.lower() or "cgm" in t.lower()):
            for s in ("val", "test"):
                now = unscale(sc, cur, S[s][cur].to_numpy())
                rows.append(dict(dataset=name, model="same-as-now", target=t, split=s, **evaluate("reg", S[s][t], now)))
        print("done", name, t)
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "baseline_metrics.csv", index=False)
print(out[out.split == "test"].round(3).to_string(index=False))
