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

from cross_validation import cross_validation2

random.seed(42)
np.random.seed(42)



train = pd.read_csv("../data/train.csv")
X_train = train.iloc[:, 2:].values
y_train = train.iloc[:, 1].values



invlogit = lambda x: 1/(1 + tt.exp(-x))
# invlogit = lambda x: (tt.tanh(x) + 1.0)/2.0

class LogisticNormalPrior:
    def __init__(self):
        return None
    
    def fit(self, X, y):
        self.model = pm.Model()
        
        with self.model:
            xi1 = pm.Bernoulli('xi1', .05, shape=X.shape[1]) # inclusion probability for each variable
            xi2 = pm.Bernoulli('xi2', .05, shape=X.shape[1]) # inclusion probability for each variable
            alpha = pm.Normal('alpha', mu=0, sd=5) # Intercept
            beta1 = pm.Normal('beta1', mu=0, sd=.75 , shape=X.shape[1]) # Prior for the non-zero coefficients
            beta2 = pm.Normal('beta2', mu=0, sd=.75 , shape=X.shape[1]) # Prior for the non-zero coefficients
            p1 = pm.math.dot(X, xi1*beta1) # Deterministic function to map the stochastics to the output
            p2 = pm.math.dot(np.power(X, 3.0), xi2*beta2)
            y_obs = pm.Bernoulli('y_obs', invlogit(alpha + p1 + p2),  observed=y)  # Data likelihood
        
        with self.model:
            self.trace = pm.sample(2000, random_seed=4816, cores=1, progressbar=False, chains=1)
        
        return None
    
    def predict(self, X):
#         test_beta1 = self.trace['beta1'][0]
#         test_beta2 = self.trace['beta2'][0]
#         test_ix1 = self.trace['xi1'][0]
#         test_ix2 = self.trace['xi2'][0]
#         test_score = expit(self.trace['alpha'][0] + np.dot(X, test_inc * test_beta))  

        estimate1 = self.trace['beta1'] * self.trace['xi1'] 
        estimate2 = self.trace['beta2'] * self.trace['xi2'] 
        y_pred = np.apply_along_axis(np.mean, 1, expit(self.trace['alpha'] + np.dot(X, np.transpose(estimate1)) + np.dot(np.power(X, 3.0), np.transpose(estimate2))))
        
        return y_pred

    
    
model = LogisticNormalPrior()
cross_validation2(X_train, y_train, model, verbose=True)



test = pd.read_csv("../data/test.csv")
X_test = test.iloc[:, 1:].values



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

submission = pd.DataFrame({"id": test["id"], "target": y_pred})
submission.to_csv("../submissions/logistic_normal_prior_cubics_"+str(int(time.time()))+".csv", index=False)