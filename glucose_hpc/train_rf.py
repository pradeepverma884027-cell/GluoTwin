"""Random Forest.   python train_rf.py --raw_dir /path/to/your/csvs [--trees 300]"""
import argparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from common import add_args, run_tabular

ap = add_args(argparse.ArgumentParser())
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--min_leaf", type=int, default=3)
a = ap.parse_args()


def make(task, pw, a):
    kw = dict(n_estimators=a.trees, min_samples_leaf=a.min_leaf, n_jobs=a.n_jobs, random_state=a.seed)
    return RandomForestRegressor(**kw) if task == "reg" else RandomForestClassifier(class_weight="balanced_subsample", **kw)


run_tabular(a, "rf", make)
