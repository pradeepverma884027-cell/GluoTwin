"""LightGBM for every dataset / target. Early stopping on val, final numbers on test.
Usage: python 02_train_lgbm.py [--datasets a b] [--targets t1 t2] [--n_jobs 16]"""
import argparse
import lightlgbm as lgb
import numpy as np, pandas as pd
from common import DATASETS, RESULTS, best_f1_threshold, evaluate, load, splits, task_of

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
ap.add_argument("--targets", nargs="*")
ap.add_argument("--n_jobs", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--leaves", type=int, default=31)
a = ap.parse_args()
(RESULTS / "models").mkdir(exist_ok=True)
rows, imps = [], []
for name in a.datasets:
    df, feats, targets, sc = load(name)
    S = splits(df)
    for t in (a.targets or targets):
        if t not in targets: continue
        task = task_of(df, t)
        P = dict(n_estimators=4000, learning_rate=a.lr, num_leaves=a.leaves, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0, random_state=42, n_jobs=a.n_jobs, verbose=-1)
        if task == "clf":
            pos = S["train"][t].mean(); P["scale_pos_weight"] = float((1 - pos) / max(pos, 1e-6))
        m = (lgb.LGBMRegressor if task == "reg" else lgb.LGBMClassifier)(**P)
        m.fit(S["train"][feats], S["train"][t], eval_set=[(S["val"][feats], S["val"][t])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = (lambda d: m.predict(d[feats])) if task == "reg" else (lambda d: m.predict_proba(d[feats])[:, 1])
        thr = best_f1_threshold(S["val"][t], pv(S["val"])) if task == "clf" else None
        for s in ("val", "test"):
            rows.append(dict(dataset=name, model="LightGBM", target=t, split=s, best_iter=m.best_iteration_,
                             **evaluate(task, S[s][t], pv(S[s]), thr)))
        imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False).head(15)
        imps += [dict(dataset=name, target=t, feature=f, gain=float(g)) for f, g in imp.items()]
        m.booster_.save_model(str(RESULTS / "models" / f"{name}__{t}.txt"))
        print("done", name, t, flush=True)
out = pd.DataFrame(rows); out.to_csv(RESULTS / "lgbm_metrics.csv", index=False)
pd.DataFrame(imps).to_csv(RESULTS / "lgbm_feature_importance.csv", index=False)
print(out[out.split == "test"].round(3).to_string(index=False))
