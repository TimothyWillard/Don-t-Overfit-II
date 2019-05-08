import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from scipy import stats
from scipy.special import expit
import theano.tensor as tt
import random
import time

from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import roc_auc_score

from cross_validation import cross_validation

random.seed(42)
np.random.seed(42)



train = pd.read_csv("../data/train.csv")
X_train = train.iloc[:, 2:].values
y_train = train.iloc[:, 1].values



# invlogit = lambda x: 1/(1 + tt.exp(-x))
invlogit = lambda x: (tt.tanh(x) + 1.0)/2.0

class LogisticNormalPrior:
    def __init__(self):
        return None
    
    def fit(self, X, y):
        self.model = pm.Model()
        
        with self.model:
            xi = pm.Bernoulli('xi', .05, shape=X.shape[1]) # inclusion probability for each variable
            alpha = pm.Normal('alpha', mu=0, sd=5) # Intercept
            beta = pm.Normal('beta', mu=0, sd=.75 , shape=X.shape[1]) # Prior for the non-zero coefficients
            p = pm.math.dot(X, xi * beta) # Deterministic function to map the stochastics to the output
            y_obs = pm.Bernoulli('y_obs', invlogit(p + alpha),  observed=y)  # Data likelihood
        
        with self.model:
            self.trace = pm.sample(2000, random_seed=4816, cores=1, progressbar=False, chains=1)
        
        return None
    
    def predict(self, X):
        test_beta = self.trace['beta'][0]
        test_inc = self.trace['xi'][0]
        test_score = expit(self.trace['alpha'][0] + np.dot(X, test_inc * test_beta))  

        estimate = self.trace['beta'] * self.trace['xi'] 
        y_pred = np.apply_along_axis(np.mean, 1, expit(self.trace['alpha'] + np.dot(X, np.transpose(estimate))))
        
        return y_pred

    
    
model = LogisticNormalPrior()
cross_validation(X_train, y_train, model, verbose=True)



test = pd.read_csv("../data/test.csv")
X_test = test.iloc[:, 1:].values



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

submission = pd.DataFrame({"id": test["id"], "target": y_pred})
submission.to_csv("../submissions/tanh_normal_prior_"+str(int(time.time()))+".csv", index=False)