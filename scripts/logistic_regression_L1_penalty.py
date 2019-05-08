import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
import time
from copy import deepcopy
import pickle as pkl

from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import roc_auc_score

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from cross_validation import cross_validation

np.random.seed(42)



train = pd.read_csv("../data/train.csv")
X_train = train.iloc[:, 2:].values
y_train = train.iloc[:, 1].values



class LogisticRegressionL1Penalty:
    method_ = "trf" # least squares method, can be `trf`, `dogbox`, `lm`
    
    def __init__(self, inclusion, alpha, loss):
        self.inclusion_ = inclusion # Array of True/False for if to include or not to include
        self.num_include_ = np.sum(inclusion) # Number of included variables
        self.alpha_ = alpha # Penalty coefficient
        self.loss_ = loss # Loss to use by lsq solver, can be `linear`, `soft_l1`, `huber`, `cauchy`, `arctan`
    
    def loss_function_(self, w):
        penalty = self.alpha_*np.linalg.norm(w, ord=1)
        rmse = np.sqrt(np.average(np.power(self.y_ - expit(np.dot(self.X_, w)) , 2.0)))
        return penalty + rmse
    
    def fit(self, X, y):
        self.X_ = deepcopy(X)
        self.X_ = self.X_[:, self.inclusion_]
        self.y_ = deepcopy(y)
        
        w_init = np.random.randn(self.num_include_)
        self.result_ = least_squares(self.loss_function_, w_init, loss=self.loss_, method=self.method_)
        self.w_ = self.result_.x
        
        return self
    
    def predict(self, X):
        return expit(np.dot(X[:, self.inclusion_], self.w_))



def model_cv_loss(params, return_cv=False):
    inclusion = params[:300]
    alpha = np.exp(params[300])
    loss = params[301]
    
    model = LogisticRegressionL1Penalty(inclusion, alpha, loss)
    mean, std = cross_validation(X_train, y_train, model, verbose=False)
    
    if return_cv:
        return mean, std
    else:
        return -mean



search_space = []
for i in range(300):
    search_space.append(Categorical([True, False]))
search_space.append(Real(np.log(1e-4), np.log(1e4)))
search_space.append(Categorical(["linear", "soft_l1", "huber", "cauchy", "arctan"]))



res_gp = gp_minimize(model_cv_loss, search_space, n_calls=300, random_state=42, verbose=True)
print(res_gp)
with open("logistic_regression_L1_penalty_search_results_"+str(int(time.time()))+".pkl", "wb") as f:
    pkl.dump(res_gp, f)
mean, std = model_cv_loss(res_gp.x, return_cv=True)
print("AUC = %.5f +/- %.5f"%(mean, std))



test = pd.read_csv("../data/test.csv")
X_test = test.iloc[:, 1:].values


model = LogisticRegressionL1Penalty(res_gp.x[:300], np.exp(res_gp.x[300]), res_gp.x[301])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

submission = pd.DataFrame({"id": test["id"], "target": y_pred})
submission.to_csv("../submissions/logistic_regression_L1_penalty_"+str(int(time.time()))+".csv", index=False)