import numpy as np
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid
import copy
import os
import json
import joblib 
from collections import Counter, OrderedDict
import logging
import math
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils.misc import list_to_dict
from utils.training import get_folds
from utils.arrays import get_subset        
from utils.arrays import apply_mask     
from utils.scoring import build_e2e
from utils.seq_prep import flatten_
from utils.misc import tuple2str, str2tuple
from models.active_learning import active_sample, SIM_TYPE_AVG, SIM_TYPE_MAX, random_split, H_TYPE_SUM, H_TYPE_LOOP, plot_label_dist
from utils.proj_setup import make_and_clear
from constants import *        

from pytorch_models.pretrained import load_pretrained


SAMPLE_TYPE_RANDOM = 'random'
SAMPLE_TYPE_ACTIVE = 'active'

def combine_multistage(scores_ss, scores_e2e):

    scores_ss[EVAL_RANGE] = SINGLE_STAGE
    scores_e2e[EVAL_RANGE] = END_TO_END
    
    # Merge single-stage and end-to-end scores
    return pd.concat([scores_ss, scores_e2e])
    

def idx_slice(X, I):
    '''
    Extract indices of I from X
    '''
    return [X[i] for i in I]            


def slice_(I, args):
    
    assert len(set([len(a) for a in args])) == 1

    out = []
    for a in args:
        out.append([a[i] for i in I])
    
    return tuple(out)

def split_(I, J, args):
    
    assert sorted(I + J) == list(range(len(args[0])))

    return (slice_(I, args), slice_(J, args))    

def idx_slice2(X, Y, I):
    '''
    Extract indices of I from X
    '''
    return (idx_slice(X, I), idx_slice(Y, I))


def get_label_dist(y):
    
    if isinstance(y, list) and isinstance(y[0], dict):
       
        # Counts event-label occurrences       
        counter = Counter()
        for dict_ in y:
            for evt_type, label in dict_.items():
                counter[(evt_type, label)] += 1

        # Convert to list of tuple
        counts = [(evt_type, label, cnt) \
                          for (evt_type, label), cnt in counter.items()]
        
        # As data frame
        df = pd.DataFrame(counts, columns=[EVENT_TYPE, LABEL, COUNT])

        return df     
        
    else:
        TypeError("Invalid y type: {} with {}".format(type(y), type(y[0])))
        
def to_json(path, file_name, X, indent=4):
    
    fn = os.path.join(path, file_name)
    with open(fn,'w') as f:
        json.dump(X, f, indent=indent)


'''
labels = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
label_pos = np.arange(len(labels))
count = [10,8,6,4,2,1]

plt.barh(label_pos, count, align='center', alpha=0.5)
plt.yticks(label_pos, labels)
plt.xlabel('Usage')
plt.title('Programming language usage')
'''


        
class Base(object):
    
    '''
    Base classifier
    '''
    
    def __init__(self,  \
        estimator_class, 
        hyperparams, 
        metric, 
        average, 
        scorer, 
        neg_label, 
        path, 
        feat_params = None, 
        event = None,
        entity = None, 
        model_type = None,
        descrip = None, 
        fit_method = 'fit', 
        prob_method = 'predict_proba', 
        pred_method = 'predict', 
        get_params_method = 'get_params',
        
        ):



        self.estimator_class = estimator_class
        self.hyperparams = hyperparams
        self.metric = metric
        self.average = average
        self.scorer = scorer
        self.neg_label = neg_label
        
        
        self.estimator = None
        
        
        self.path = path
        self.feat_params = feat_params
        self.event = event
        self.entity = entity
        self.model_type = model_type
        self.descrip = descrip

        self.fit_method = fit_method
        self.prob_method = prob_method
        self.pred_method = pred_method
        self.get_params_method = get_params_method


    def save_state_dict(self, path):
        '''
        Save state dict
        '''
       
        state_dict = self.estimator.state_dict()
        
        if hasattr(self.estimator, 'state_dict_exclusions'):
            for excl in self.estimator.state_dict_exclusions:
                if excl in state_dict:
                    del state_dict[excl]
       
        torch.save(state_dict, path)
        return True 
        

    #    def load_state_dict(self, path):
    #        '''
    #        Load state dict
    #        '''
    #        self.estimator = self.estimator_class(**self.hyperparams)
    #        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    #        self.estimator.load_state_dict(state_dict)
    #        return True
    #
    #    def load_pretrained(self, dir_, param_map=None):
    #    
    #        # Hyperparameters
    #        logging.info('')
    #        logging.info('-'*72)
    #        logging.info('Loading pre-trained model...')
    #        logging.info('-'*72)
    #        
    #        # Load hyper parameters
    #        fn = os.path.join(dir_, HYPERPARAMS_FILE)
    #        logging.info("Hyper parameters loading from:\t{}".format(fn))
    #        hp = json.load(open(fn,'r'))
    #        
    #        # Print hyper parameters
    #        logging.info("Hyper parameters loaded:")
    #        for param, val in hp.items():
    #            logging.info("\t{}:\t{}".format(param, val))
    #        logging.info('')
    #
    #        # Map parameters
    #        if param_map is not None:
    #            logging.info('Mapping hyper parameters:')
    #            for name, val in param_map.items():
    #                logging.info('\t{}: orig={},\tnew={}'.format(name, hp[name], val))
    #                hp[name] = val
    #
    #        # Incorporate hyper parameters into model
    #        self.hyperparams = hp
    #        logging.info('self.hyperparams updated')
    #        
    #        # Load saved estimator
    #        fn = os.path.join(dir_, STATE_DICT)
    #        logging.info("State dict loading from:\t{}".format(fn))
    #        self.load_state_dict(fn)
    #        logging.info("State dict loaded")
    #        
    #        logging.info('Pre-trained model loaded')
    #        logging.info('')
    #        logging.info('')
    #        
    #        return True


    def load_pretrained(self, model_dir, param_map=None):

        self.estimator, self.hyperparams = load_pretrained( \
                                model_class = self.estimator_class,
                                model_dir = model_dir, 
                                param_map = param_map)
        return True

    def fit_preprocess_X(self, X):
        '''
        Placeholder for fitting preprocessing function for features
        '''
        pass
        

    def fit_preprocess_y(self, y):
        '''
        Placeholder for fitting preprocessing function for labels
        '''
        pass

    def preprocess(self, X, y):
        '''
        Placeholder for preprocessing features and labels
        (e.g. convert feature dictionaries to one hot encodings)
        '''
        return (X, y)

    
    def postprocess(self, X, y):
        '''
        Placeholder for postprocessing features and labels

        '''
        return (X, y)

    def postprocess_prob(self, X, y_prob):
        '''
        Placeholder for postprocessing features and probabilities

        '''
        return y_prob

    def apply_mask(self, X, mask=None):
        '''
        Apply mask to input, returning masked to version
        '''
        
        # If no mask provide, include all
        if mask is None:
            return X
        
        # Return none if X None
        if X is None:
            return X
        
        # Use mask to extract subset of features and labels
        else:        
            return apply_mask(X, mask)

    def get_default_prob(self, y_prob):
        '''
        Get default probability, based on probably predictions
        '''

        # Set all label probabilities to 0
        d = {lab:0.0 for lab, prob in y_prob[0].items()}
            
        # Assign all probability mass to negative label
        d[self.neg_label] = 1.0
        
        return d
        
    def restore_mask(self, X_e2e, y_ss, mask, default_val=None):
        '''
        Restore (unmask) predictions
        '''        
        
        # Update default value if not provided
        if default_val is None:
            default_val = self.neg_label
        
        # If no mask provide, assume all predictions present
        if mask is None:
            mask = [1]*len(X_e2e)
        
        # Get end-to-end labels
        y_e2e = build_e2e(X_e2e, y_ss, mask, default_val)
        
        return y_e2e
        

    def post_train(self):
        '''
        Placeholder for any post training actions
        '''
        return True 

    def dump(self, directory, fn):
        '''
        Save classifier to directory, with default model name
        '''
        
        # Save classifier 
        joblib.dump(self, os.path.join(directory, fn))


    def save_predictions(self, path, y_pred):

        fn = os.path.join(path, PREDICTIONS_FILE)
        joblib.dump(y_pred, fn)
        
        return True

    def save_scores(self, path, scores):
       
       
        if isinstance(scores, dict):

            for k, df in scores.items():

                # Save scores delimited text file
                fn = os.path.join(path, 'scores_{}.csv'.format(k))
                df.to_csv(fn)
                            
        
        else:       
       
            # Save scores delimited text file
            fn = os.path.join(path, SCORES_FILE)
            scores.to_csv(fn, index=False)
        
        return True


    def save_sweeps(self, path, sweeps):

        if sweeps is not None:
            fn = os.path.join(path, SWEEPS_FILE)
            sweeps.to_csv(fn, index=True)
        
        return True        


    def save_hyperparams(self, path, params):
        fn = os.path.join(path, HYPERPARAMS_FILE)
        with open(fn, 'w') as f:
            json.dump(params, f)

    def save_featparams(self, path, params):
        fn = os.path.join(path, FEATPARAMS_FILE)
        with open(fn, 'w') as f:
            json.dump(params, f)        

    def save_descrip(self, path, descrip):
        fn = os.path.join(path, DESCRIP_FILE)
        with open(fn, 'w') as f:
            json.dump(descrip, f)


    def save_results(self, path, y_pred, scores, sweeps=None):
        '''
        Save results to disk
        '''    
       
        # Save predictions
        self.save_predictions(path, y_pred)
        
        # Save scores
        self.save_scores(path, scores)
        
        # Save sweeps as spreadsheet
        self.save_sweeps(path, sweeps)
        
        return True 


    def fit(self, X, y, mask=None, post_train=True, **kwargs):
        '''
        Fit estimator to data
        '''

        # Apply mask
        X = self.apply_mask(X, mask)
        y = self.apply_mask(y, mask)

        # Preprocess X and y
        X, y = self.preprocess(X, y)

        # Estimator not initialized
        if self.estimator is None:
            self.estimator = self.estimator_class(**self.hyperparams)

        # Fit to data
        self.estimator.fit(X, y, **kwargs)

        # Post training actions
        if post_train:
            self.post_train()      
        
        return True 


    def predict_sub_op(self, X, **kwargs):
        '''
        Predict labels, without any masking
        '''
        
        # Preprocess X (y not available)
        X, _ = self.preprocess(X, None)
        
        # Get predictions
        y = self.estimator.predict(X, **kwargs)
        
        # Postprocess predictions
        _, y = self.postprocess(None, y)

        return y 
        
    def prob_sub_op(self, X):
        '''
        Predict probabilities, without any masking
        '''
        
        # Preprocess X (y not available)
        X, _ = self.preprocess(X, None)
        
        # Get probability predictions
        y = self.estimator.proba(X)                
        
        # Postprocess probability predictions
        _, y = self.postprocess_prob(None, y)

        return y

    def predict(self, X, mask=None, **kwargs):
        '''
        Predict labels, potentially with masking
        '''
        
        # Apply mask
        X_ss = self.apply_mask(X, mask)        
        
        # Get predictions
        y_ss = self.predict_sub_op(X_ss, **kwargs)
        
        # Restore/unmask
        if mask is None:
            return y_ss
        else:
            y_e2e = self.restore_mask( \
                                        X_e2e = X, 
                                        y_ss = None, 
                                        mask = mask)
            return y_e2e           

    def predict_proba(self, X, mask=None):
        '''
        Predict label probability, potentially with masking
        '''
        
        # Apply mask
        X_ss = self.apply_mask(X, mask)        
        
        # Get predictions
        y_prob_ss = self.prob_sub_op(X_ss)
                
        ## Get default probability
        #default_val = self.get_default_prob(y_prob_ss)
        
        # Restore/unmask
        #y_prob_e2e = self.restore_mask( \
        #                                X_e2e = X, 
        #                                y_ss = y_prob_ss, 
        #                                mask = mask,
        #                                default_val = default_val)
        x = ss               
        #return y_prob_e2e 
        return y_prob_ss


    def get_params(self):
        params = {}
        params.update(self.hyperparams)
        if self.feat_params is not None:
            params.update(self.feat_params)
        params.update({'model_type': self.model_type,
                      'descrip': self.descrip}

        #params.update({'event': self.event, 
        #              'entity': self.entity}

        )
        return params

    def score(self, X, y, mask=None, path=None, **kwargs):
        '''
        Predict labels and score result
        '''

        # Preprocess labels
        _, y = self.preprocess(None, y)
        
        # Apply mask
        X_ss = self.apply_mask(X, mask)
        y_ss = self.apply_mask(y, mask)     
                
        # Get predictions
        y_pred_ss = self.predict_sub_op(X_ss, **kwargs)
               
        # Score single state results
        scores_ss = self.scorer.fit(y_ss, y_pred_ss, \
                                               params=self.get_params())

        if mask is None:

            # Save results
            if path is not None:
                self.save_results(path, y_pred_ss, scores_ss, None)
            
            return (y_pred_ss, scores_ss)

        else:            

            # Restore/unmask
            y_pred_e2e = self.restore_mask(X, y_pred_ss, mask = mask)
            
            # Evaluate end-to-end performance
            scores_e2e = self.scorer.fit(y, y_pred_e2e, params=self.get_params())
            
            # Merge single-stage and end-to-end scores
            scores = combine_multistage(scores_ss, scores_e2e)

            # Save results
            if path is not None:
                self.save_results(path, y_pred_e2e, scores, None)
               
            return (y_pred_e2e, scores)
        

    def cv_predict(self, X, y, cv, \
                tune = False, \
                params = None, 
                mask = None,
                retrain = False,
                path = None,
                **kwargs):    
        '''
        Run estimator to make predictions 
        
        args:
            tune: Boolean, True = tune hyper parameters
                           False = do not tune, just predict
            
        returns:
            truth, predictions, and label probabilities
                    
        '''

        # Model hyperparams
        hyperparams = {}
        for k, v in self.hyperparams.items():
            hyperparams[k] = v
        if params is not None:
            hyperparams.update(params)

        # Apply mask
        X_ss = self.apply_mask(X, mask)
        y_ss = self.apply_mask(y, mask)

        # Create folds
        folds = get_folds(X_ss, y_ss, cv, tune)
                              
        # Loop on folds
        y_true_ss = []
        y_pred_ss = []        
        for i, (X_fit, y_fit, X_eval, y_eval) in enumerate(folds):

            # Fit to folds
            self.estimator = self.estimator_class(**hyperparams)
            self.fit(X_fit, y_fit, mask=None, post_train=False, **kwargs)            

            # Aggregate results
            y_pred_ss.extend(self.predict_sub_op(X_eval, **kwargs)) 
            y_true_ss.extend(y_eval)
        
        # Get an update parameters
        params_tmp = copy.deepcopy(self.get_params())
        params_tmp.update(hyperparams)

        # Preprocess labels
        _, y_true_ss = self.preprocess(None, y_true_ss)

        # Score result
        scores_ss = self.scorer.fit(y_true_ss, y_pred_ss, params=params_tmp)       

        # If not tune, then all records include predictions
        #   so determine end-to-end results.
        if (mask is not None) and (not tune):
            
            # Restore/unmask
            y_pred_e2e = self.restore_mask(X, y_pred_ss, mask=mask)
            _, y_true_e2e = self.preprocess(None, y)
            
            # Evaluate end-to-end performance
            scores_e2e = self.scorer.fit(y_true_e2e, y_pred_e2e, params=params_tmp)
            
            # Merge single-stage and end-to-end scores
            scores = combine_multistage(scores_ss, scores_e2e)
        
        # Only return single-state results
        else:
            scores = scores_ss
            y_pred_e2e = y_pred_ss
        
        # Re-train model with best parameters
        if retrain:  
            
            # Update model parameters      
            self.hyperparams = hyperparams
            
            # Fit model
            self.estimator = self.estimator_class(**self.hyperparams)
            self.fit(X_ss, y_ss, mask = None, post_train = True, **kwargs)            

        # Save results
        if path is not None:
            self.save_results(path, y_pred_e2e, scores, None)
            
        return (y_pred_e2e, scores)

    def cv_tune(self, X, y, cv, param_sweep,
                tune = False, \
                params = None, 
                mask = None,
                retrain = False,
                path = None):    
        '''
        Cross-validation runs for tuning hyper parameters
        '''

        # Create exhaustive parameter grid
        sweep_params = list(ParameterGrid(param_sweep))

        # Initialize output score vector
        sweeps = np.zeros((len(sweep_params),))-1       

        # Loop on parameter combos
        all_scores = []
        all_hyperparams = []
        for param_idx, params in enumerate(sweep_params):

            # Parameters for current run           
            current_params = self.hyperparams.copy()
            current_params.update(params)
            
            # Get performance with current parameters
            y_pred, scores = self.cv_predict(X, y, cv, \
                        tune = True, \
                        params = current_params, 
                        mask = mask,
                        retrain = False,
                        path = None)
            
            all_scores.append(scores)
            all_hyperparams.append(current_params)
            
            # Extract evaluation metric 
            sweeps[param_idx] = \
                 scores.loc[scores[LABEL] == self.average][self.metric].values[0]
               
        # Best param set
        idx = np.argmax(sweeps)
        best_param = all_hyperparams[idx]
        best_scores = all_scores[idx]

        # Parameters sweep results per fold, as dataframe
        col = ['{}_{}'.format(self.average, self.metric)]
        dfscores = pd.DataFrame(sweeps, columns=col)       
        dfparams = pd.DataFrame(sweep_params)
        dfsweeps = pd.concat([dfparams, dfscores], axis=1)

        # Save results
        if path:
            
            # Save scores
            self.save_scores(path, best_scores)
        
            # Save sweeps as spreadsheet
            self.save_sweeps(path, dfsweeps)            
            
            # Save hyper and feature parameters
            self.save_hyperparams(path, best_param)
            self.save_featparams(path, self.feat_params)
            self.save_descrip(path, self.descrip)


        return (best_param, best_scores, dfsweeps)


    def active_learn(self, \
        X_pool, 
        y_pool, 
        i_pool,
        X_eval, 
        y_eval, 
        i_eval, 
        hyperparams, 
        X_init = None,
        y_init = None,
        i_init = None, 
        model_dir_init = None,
        n_init = 100,
        n_batch = 50,
        n_batches = 1,
        sample_type = SAMPLE_TYPE_RANDOM,
        embed_pool = None,
        entropy_type = H_TYPE_LOOP,
        sim_type = SIM_TYPE_MAX,
        alpha = 1.0,
        path = None,
        col_val_filt = None,
        metric = None,
        seeds = None,
        param_map = None):

        logging.info("")
        logging.info("Active learning training")

        # Consolidate subset of parameters
        params = {}        
        params['n_init'] = n_init if X_init is None else len(X_init)
        params['n_batch'] = n_batch
        params['sample_type'] = sample_type
        params['entropy_type'] = entropy_type
        params['sim_type'] = sim_type
        params['alpha'] = alpha
        params['col_val_filt'] = col_val_filt
        params['metric'] = metric
        params.update(hyperparams)
        
        
        # Check lengths
        assert len(set([len(X_pool), len(y_pool), len(i_pool)]))==1
        assert len(set([len(X_eval), len(y_eval), len(i_eval)]))==1
        if X_init is not None:
            assert len(set([len(X_init), len(y_init), len(i_init)]))==1
        assert (embed_pool is None) or (len(embed_pool) == len(X_pool))

        # Get data set sizes
        logging.info('') 
        logging.info('Input sizes:')
        logging.info("Pool size:\t{}".format(len(X_pool)))
        logging.info("Eval size:\t{}".format(len(X_eval)))
        logging.info("Initial size:\t{}".format(None if X_init is None else len(X_init)))

        # Get next seed
        seed = None if seeds is None else seeds.pop()
            
        # Initial training data not provided, sample from training pool
        logging.info('')
        logging.info('Initial batch set up')
        if X_init is None:

            logging.info('Initial training data NOT provided.')
            logging.info('Sampling initial training data from training pool.')
            logging.info("Initial training size:\t{}".format(n_init))

            # Start with indices for all training data
            idx_pool = list(range(len(X_pool)))

            # Randomly sample initial training data from pool
            idx_init, _, idx_pool, _ = random_split(idx_pool, n_init, seed=seed)
            
            # Initial and pool subsets
            X_init, y_init, i_init = slice_(idx_init, (X_pool, y_pool, i_pool))
            X_pool, y_pool, i_pool = slice_(idx_pool, (X_pool, y_pool, i_pool))
            
        # Initial training data provided
        else:            
            n_init = len(X_init)
            logging.info('Initial training data provided')
            logging.info('Overriding provided n_init value')
            logging.info("Init size:\t{}".format(n_init))

        # Get data set sizes
        n_pool = len(X_pool)
        n_eval = len(X_eval) 
        n_init = len(X_init)     
        logging.info('') 
        logging.info('Input sizes after setup:')
        logging.info("Pool size:\t{}".format(n_pool))
        logging.info("Eval size:\t{}".format(n_eval))
        logging.info("Init size:\t{}".format(n_init))

        # Total training samples
        n_train = n_init + n_pool
            
        # Create round-specific directory
        path_ = os.path.join(path, 'round_{}'.format(0))
        make_and_clear(path_)

        # Record IDs
        to_json(path_, 'ids_pool.json', i_pool)
        to_json(path_, 'ids_eval.json', i_eval)
        to_json(path_, 'ids_init.json', i_init)
       
        '''
        Iterate over training batches
        '''
       
        # Set current training data to initial training data
        X_fit = X_init[:]
        y_fit = y_init[:]
        i_fit = i_init[:]
        i_fit_all = OrderedDict()
        # Loop on training batches   
        scores = []
        dfs_labels = []
        for i in range(n_batches + 1):


            assert n_train == len(X_fit) + len(X_pool)
            assert len(set([len(X_fit),  len(y_fit),  len(i_fit) ]))==1            
            assert len(set([len(X_pool), len(y_pool), len(i_pool)]))==1

            # Indices of pool
            idx_pool = list(range(len(X_pool)))

            # Create round-specific directory
            path_ = os.path.join(path, 'round_{}'.format(i))
            make_and_clear(path_)
            
            # Initial round for initial training
            if i == 0:
                idx_batch = []
                idx_pool = idx_pool


            # Random selection
            elif sample_type == SAMPLE_TYPE_RANDOM:
              
                seed = None if seeds is None else seeds.pop()
                idx_batch, _, idx_pool, _ = \
                            random_split(idx_pool, n_batch, seed=seed)
                fn = os.path.join(path_, 'random_sampling.txt')
                with open(fn,'w') as f:
                    f.write('{} random samples drawn uisng seed={}'.format(n_batch, seed))
                
            # Active selection    
            elif sample_type == SAMPLE_TYPE_ACTIVE:
                
                # Get relevant embeddings                
                
                # Generate entropy scores
                prob_pool, entropy_pool = self.estimator.entropy(X_pool)
                
                assert len(X_pool) == len(embed_pool)
                assert len(X_pool) == len(entropy_pool)

                # Select batch
                idx_batch, idx_pool = active_sample( \
                                            sample_count = n_batch, 
                                            ids = idx_pool, 
                                            embed = embed_pool, 
                                            entropy = entropy_pool, 
                                            entropy_type = entropy_type,
                                            sim_type = sim_type,
                                            alpha = alpha,
                                            path = path_,
                                            docs = X_pool,
                                            prob = prob_pool)
                embed_pool, = slice_(idx_pool, (embed_pool,))
                
            else:
                raise ValueError("Invalid sample_type:\t{}".format(sample_type))


            assert n_train == len(X_fit) + len(X_pool)
            assert len(set([len(X_fit),  len(y_fit),  len(i_fit) ]))==1            
            assert len(set([len(X_pool), len(y_pool), len(i_pool)]))==1


            # Separate batch and remaining pool
            X_batch, y_batch, i_batch = slice_(idx_batch, (X_pool, y_pool, i_pool))
            X_pool,  y_pool,  i_pool  = slice_(idx_pool,  (X_pool, y_pool, i_pool))
            
            # Add training data to fit
            X_fit.extend(X_batch)
            y_fit.extend(y_batch)
            i_fit.extend(i_batch)
            i_fit_all[i] = i_fit[:]
            
            # Get label distribution for selected batch            
            df = get_label_dist(y_fit)
            df[ROUND] = i
            dfs_labels.append(df)

            logging.info("")
            logging.info("="*72)
            logging.info("Training batch:\t\t{} of {}".format(i+1, n_batches))
            logging.info("="*72)
            logging.info("Samples in batch:\t{}".format(len(X_batch)))
            logging.info("Samples in train:\t{}".format(len(X_fit)))
            logging.info("Samples in pool:\t{}".format(len(X_pool)))
            
                    
            # Fit model to batch

            # Initial round for initial training
            if (i == 0) and (model_dir_init is not None):
                #self.load_pretrained(dir_=model_dir_init, param_map=param_map)
                self.load_pretrained( \
                                    model_dir = model_dir_init, 
                                    param_map = param_map)
                
                
                logging.info('')
                logging.info('Loading initial model')
                logging.info('')
                

                fn = os.path.join(path_, 'loaded pre-trained.txt')
                with open(fn,'w') as f:
                    f.write('Loaded pre-trained model from: {}'.format(model_dir_init))

                
            else:                            
                self.estimator = self.estimator_class(**hyperparams)
                self.fit(X_fit, y_fit, mask=None, post_train=False)     

            # Get predictions
            y_pred = self.predict(X_eval)
            _, y_true = self.preprocess(None, y_eval)

            # Score result
            s = self.scorer.fit(y_true, y_pred, params=params)      
            s['train count'] = len(X_fit)
            s['corpus fraction'] = len(X_fit)/n_train
            s['round'] = i
            scores.append(s)

            assert n_train == len(X_fit) + len(X_pool)
            assert len(set([len(X_fit),  len(y_fit),  len(i_fit) ]))==1            
            assert len(set([len(X_pool), len(y_pool), len(i_pool)]))==1
       
        to_json(path, 'ids_fit.json', i_fit_all)

        scores = pd.concat(scores)
        fn = os.path.join(path, SCORES_FILE)
        scores.to_csv(fn)

        # Aggregate label distributions
        dfs_labels = pd.concat(dfs_labels)
        fn = os.path.join(path, 'label_dist.png')
        plot_label_dist(dfs_labels, fn)
        fn = os.path.join(path, 'label_dist.csv')
        dfs_labels.to_csv(fn)
        
        
        if (col_val_filt is not None) and (metric is not None):
            summary = scores.copy()
            for col, val in col_val_filt:
                summary = summary[summary[col]==val]
                
                
            fn = os.path.join(path, 'summary.csv')
            summary.to_csv(fn)

            plot = summary.plot(x ='corpus fraction', y=metric, kind='bar')

            # Save plot
            fig = plot.get_figure()
            fn = os.path.join(path, 'summary.png')
            fig.savefig(fn)
            plt.close()
    
        return scores


