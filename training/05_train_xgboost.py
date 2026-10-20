"""Train XGBoost separately. Requires xgboost."""
import xgboost as xgb
from common import RANDOM_STATE, run_model


def factory(task, y):
    params = dict(n_estimators=3000, learning_rate=0.03, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
                  early_stopping_rounds=100)
    if task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", **params)
    positive = max(float(y.mean()), 1e-6)
    params["scale_pos_weight"] = (1.0 - positive) / positive
    return xgb.XGBClassifier(eval_metric="logloss", **params)


def fit(model, Xtr, ytr, Xval, yval, task):
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

if __name__ == "__main__":
    run_model("xgboost", factory, early_stopping_fit=fit)
