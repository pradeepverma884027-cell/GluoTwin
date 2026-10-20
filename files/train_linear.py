"""Ridge regression / Logistic regression (+ "same as now" rule). Always run this first as your baseline.
python train_linear.py --raw_dir /path/to/your/csvs"""
import argparse
from sklearn.linear_model import LogisticRegression, Ridge
from common import add_args, run_tabular

ap = add_args(argparse.ArgumentParser())
ap.add_argument("--alpha", type=float, default=1.0)
a = ap.parse_args()
run_tabular(a, "linear",
            lambda task, pw, a: Ridge(alpha=a.alpha) if task == "reg"
            else LogisticRegression(max_iter=3000, class_weight="balanced"),
            with_same_as_now=True)
