"""LightGBM with early stopping on the val split.   python train_lgbm.py --raw_dir /path/to/your/csvs"""
import argparse
import lightgbm as lgb
from common import add_args, run_tabular

ap = add_args(argparse.ArgumentParser())
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--leaves", type=int, default=31)
a = ap.parse_args()


def make(task, pw, a):
    kw = dict(n_estimators=4000, learning_rate=a.lr, num_leaves=a.leaves, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=a.seed, n_jobs=a.n_jobs, verbose=-1)
    return lgb.LGBMRegressor(**kw) if task == "reg" else lgb.LGBMClassifier(scale_pos_weight=pw, **kw)


def fit(m, Xtr, ytr, Xva, yva):
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])


run_tabular(a, "lgbm", make, fit)
