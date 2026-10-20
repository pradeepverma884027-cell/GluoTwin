"""Train LightGBM separately. Install lightgbm first; validation set is used for early stopping."""
import lightgbm as lgb
from common import RANDOM_STATE, run_model


def factory(task, y):
    params = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    if task == "regression":
        return lgb.LGBMRegressor(**params)
    positive = max(float(y.mean()), 1e-6)
    params["scale_pos_weight"] = (1.0 - positive) / positive
    return lgb.LGBMClassifier(**params)


def fit(model, Xtr, ytr, Xval, yval, task):
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])

if __name__ == "__main__":
    run_model("lightgbm", factory, early_stopping_fit=fit)
