import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import time
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from copy import deepcopy
from cross_validation import cross_validation
import pickle as pkl

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



def model_cv_loss(params, return_cv=False):
    gamma = np.exp(params[0])
    C = np.exp(params[1])
    
    svm_classifier = SVC(kernel="rbf", gamma=gamma, C=C, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=42)
    
    mean, std = cross_validation(X_quantile_train, y_train, svm_classifier, verbose=False, return_scores=False)
    
    if return_cv:
        return mean, std
    else:
        return -mean
    
search_space = [
    Real(np.log(1e-4), np.log(1e4)),
    Real(np.log(1e-4), np.log(1e4))
]

res_gp = gp_minimize(model_cv_loss, search_space, n_calls=300, random_state=42, verbose=True)
print(res_gp)
with open("simple_svm_classifier_search_results_"+str(int(time.time()))+".pkl", "wb") as f:
    pkl.dump(res_gp, f)
mean, std = model_cv_loss(res_gp.x, return_cv=True)
print("AUC = %.5f +/- %.5f"%(mean, std))



svm_classifier = SVC(kernel="rbf", gamma=np.exp(res_gp.x[0]), C=np.exp(res_gp.x[1]), shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=42, probability=True)
svm_classifier.fit(X_quantile_train, y_train)

y_pred = svm_classifier.predict(X_quantile_test)
submission_df = pd.DataFrame(data={"id":test_id, "target":y_pred})
submission_df.to_csv("../submissions/simple_svm_classifier_hard_prediction_"+str(int(time.time()))+".csv", index=False)

y_pred = svm_classifier.predict_proba(X_quantile_test)[:, 1]
submission_df = pd.DataFrame(data={"id":test_id, "target":y_pred})
submission_df.to_csv("../submissions/simple_svm_classifier_soft_prediction_"+str(int(time.time()))+".csv", index=False)