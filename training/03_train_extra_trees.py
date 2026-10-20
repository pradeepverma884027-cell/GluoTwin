"""Train an Extra Trees model separately for comparison."""
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from common import RANDOM_STATE, run_model

N_ESTIMATORS = 500

def factory(task, y):
    params = dict(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
                  min_samples_leaf=2, max_features=0.8)
    if task == "regression":
        return ExtraTreesRegressor(max_depth=None, **params)
    return ExtraTreesClassifier(class_weight="balanced", **params)

if __name__ == "__main__":
    run_model("extra_trees", factory)
