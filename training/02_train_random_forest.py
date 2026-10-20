"""Train a Random Forest model separately."""
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from common import RANDOM_STATE, run_model

N_ESTIMATORS = 500

def factory(task, y):
    params = dict(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
                  min_samples_leaf=2, max_features=0.8)
    if task == "regression":
        return RandomForestRegressor(max_depth=30, **params)
    return RandomForestClassifier(class_weight="balanced_subsample", **params)

if __name__ == "__main__":
    run_model("random_forest", factory)
