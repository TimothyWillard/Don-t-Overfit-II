import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import time



random.seed(42)
np.random.seed(42)



train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")



y_train = train_df["target"].values
X_train = train_df.iloc[:, 2:].values
train_id = train_df["id"].values
X_test = test_df.iloc[:, 1:].values
test_id = test_df["id"].values



quantile_transformer = QuantileTransformer(n_quantiles=1000, output_distribution="normal", ignore_implicit_zeros=False, subsample=100000, random_state=42, copy=True)
quantile_transformer.fit(np.vstack([X_train, X_test]))
X_quantile_train = quantile_transformer.transform(X_train)
X_quantile_test = quantile_transformer.transform(X_test)



logistic_regression = LogisticRegression(dual=False, tol=0.001, random_state=42, solver="liblinear", max_iter=100, multi_class="ovr", verbose=False, warm_start=False, n_jobs=1)
repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
search_grid = {
    "penalty": ["l1", "l2"],
    "C": np.geomspace(1e-3, 1e3, 50),
    "fit_intercept": [True, False],
    "intercept_scaling": np.geomspace(1e-2, 1e2, 15),
    "class_weight": [None, "balanced"],
}
grid_search = GridSearchCV(logistic_regression, search_grid, scoring="roc_auc", n_jobs=1, iid=True, refit=True, cv=repeated_kfold, verbose=True, error_score=0.0, return_train_score=False)
grid_search.fit(X_quantile_train, y_train)



cv_results_df = pd.DataFrame(data=grid_search.cv_results_)
cv_results_df.to_csv("../cv_results/simple_logistic_regression_"+str(time.time())+".csv")



y_pred = grid_search.predict(X_quantile_test)
submission_df = pd.DataFrame(data={"id":test_id, "target":y_pred})
submission_df.to_csv("../submissions/simple_logistic_regression_"+str(time.time())+".csv", index=False)