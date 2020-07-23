import pandas as pd
import numpy as np
import copy


import json
from utils.arrays import get_subset


from collections import Counter
import pandas as pd


def get_fold_indices(n, cv):
    '''
    Indices of folds
        Creates list of tuple (train idx, tune idx, test idx) for each
        fold
        
    '''
    
    # Check cross validation count
    assert cv >= 3, 'cv must be >= 0, provided cv = {}'.format(cv)
    
    # Get indices of X in states
    get_indices = lambda X_, states: np.where(np.in1d(X_, states))[0]
        
    # Folds
    folds = np.arange(cv)
        
    # Create fold assignments
    step = int(n/float(cv))
    fold_assignments = [min(int(i/float(step)), cv-1) \
                                                for i in range(0, n)]

    # Iterate over folds
    fold_indices = []
    for i, fold in enumerate(folds):
        
        # Shift (roll) folds
        rolled = np.roll(folds, -i)
        
        # Fold indices for training, tuning, and testing
        test = rolled[0] 
        tune = rolled[1]
        train = rolled[2:]
        
        # Data indices for training, tuning, and testing
        train_ind = get_indices(fold_assignments, train)
        tune_ind = get_indices(fold_assignments, tune)
        test_ind = get_indices(fold_assignments, test)
        
        # Double check fold assignments
        overlap1 = set(train_ind) & set(tune_ind)
        overlap2 = set(tune_ind) & set(test_ind)
        overlap_check = (len(overlap1) == 0) and (len(overlap2) == 0)
        length_folds = len(train_ind) + len(tune_ind) + len(test_ind)
        length_check = length_folds == n
        assert overlap_check and length_check, "Error assigning folds"
        
        # Build output as list of tuple
        fold_indices.append((train_ind, tune_ind, test_ind))
    
    return fold_indices


def get_folds(X, y, cv, tune):
    '''
    Generate folds
    
    args:
          tune: Boolean, True = tune hyper parameters
                         False = do not tune, just predict
    '''

    assert len(X) == len(y), "Length error"

    # Create folds
    n = len([True for x in X])
    fold_indices = get_fold_indices(n, cv)

    # Update order
    if tune:
        fold_indices = [fold_indices[-1]] + fold_indices[0:-1]
                          
    # Loop on folds
    folds = []
    for train_idx, tune_idx, test_idx in fold_indices:
        
        # Get training subsets
        if tune:
            fit_idx = train_idx
            eval_idx = tune_idx

        else:
            
            fit_idx = np.concatenate((train_idx, tune_idx), axis=0)
            eval_idx = test_idx
            
        # Get training/fit subsets    
        X_fit = get_subset(X, fit_idx)
        y_fit = get_subset(y, fit_idx)

        # Get evaluation subsets
        X_eval = get_subset(X, eval_idx)
        y_eval = get_subset(y, eval_idx)
            
        yield (X_fit, y_fit, X_eval, y_eval)


def get_round_indices(n_samples, n_round):
    '''
    Generate indices for multi-round training (i.e. active learning)
        Creates list of list of training indices

    Parameters
    ----------
    
        
    '''



    batch_indices = []
    for i in range(0, n_samples, n_round):
        batch_indices.append(list(range(i, i + n_round)))
    print(sum([len(X) for X in batch_indices]), n_samples)
    assert sum([len(X) for X in batch_indices]) == n_samples
    

    print(batch_indices)    

    x = laskjdf    
    return fold_indices