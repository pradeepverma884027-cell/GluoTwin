"""Train Ridge Regression or Logistic Regression, selected automatically by target type."""
from sklearn.linear_model import Ridge, LogisticRegression
from common import RANDOM_STATE, run_model

def factory(task, y):
    if task == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)

if __name__ == "__main__":
    run_model("linear", factory)
