"""XGBoost with early stopping on the val split.   python train_xgb.py --raw_dir /path/to/your/csvs [--gpu]"""
import argparse
import xgboost as xgb
from common import add_args, run_tabular

ap = add_args(argparse.ArgumentParser())
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--gpu", action="store_true")
a = ap.parse_args()


def make(task, pw, a):
    kw = dict(n_estimators=4000, learning_rate=a.lr, max_depth=a.depth, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", device="cuda" if a.gpu else "cpu", early_stopping_rounds=100,
              n_jobs=a.n_jobs, random_state=a.seed)
    return xgb.XGBRegressor(**kw) if task == "reg" else xgb.XGBClassifier(scale_pos_weight=pw, **kw)


def fit(m, Xtr, ytr, Xva, yva):
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


run_tabular(a, "xgb", make, fit)
