import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

import time
from copy import deepcopy

def cross_validation(X, y, model, verbose=False, return_scores=False):
    repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
    counter = 1
    scores = []
    
    for train_ix, valid_ix in repeated_kfold.split(X):
        start_time = time.time()
        if verbose:
            print("Starting Iteration "+str(counter)+" at "+str(int(start_time))+"s")
        
        X_train = X[train_ix]
        X_valid = X[valid_ix]
        y_train = y[train_ix]
        y_valid = y[valid_ix]
        
        sample_model = deepcopy(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        scores.append(roc_auc_score(y_valid, y_pred))
        
        end_time = time.time()
        dt = end_time - start_time
        if verbose:
            print("Finishing Iteration "+str(counter)+" at "+str(int(end_time))+"s")
            print("Finished Iteration "+str(counter)+" in "+str(int(dt))+"s")
        counter += 1

    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    
    if verbose:
        print("AUC = %.5f +/- %.5f"%(scores_mean, scores_std))    
    
    if return_scores:
        return scores
    else:
        return scores_mean, scores_std
    
def cross_validation2(X, y, model, verbose=False, return_scores=False):
    repeated_kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    counter = 1
    scores = []
    
    for train_ix, valid_ix in repeated_kfold.split(X, y):
        start_time = time.time()
        if verbose:
            print("Starting Iteration "+str(counter)+" at "+str(int(start_time))+"s")
        
        X_train = X[train_ix]
        X_valid = X[valid_ix]
        y_train = y[train_ix]
        y_valid = y[valid_ix]
        
        sample_model = deepcopy(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        scores.append(roc_auc_score(y_valid, y_pred))
        
        end_time = time.time()
        dt = end_time - start_time
        if verbose:
            print("Finishing Iteration "+str(counter)+" at "+str(int(end_time))+"s")
            print("Finished Iteration "+str(counter)+" in "+str(int(dt))+"s")
        counter += 1

    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    
    if verbose:
        print("AUC = %.5f +/- %.5f"%(scores_mean, scores_std))    
    
    if return_scores:
        return scores
    else:
        return scores_mean, scores_std