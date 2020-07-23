from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import re

import numpy as np
import pandas as pd
from six import iteritems
from collections import OrderedDict

import copy
from utils.arrays import apply_mask
from utils.seq_prep import flatten_
from utils.misc import list_to_dict, list_to_nested_dict
from utils.seq_prep import preprocess_labels_seq, postprocess_labels_seq
#from corpus.event import BIO_to_span_doc, Span


from constants import *


#COLUMNS = ['TP', 'FN', 'TN', 'FP',
#                   'precision_def', 'recall_def', 'f1_def', 
#                   'precision_opt', 'recall_opt', 'f1_opt', 
#                   'prc_auc', 'roc_auc']


'''
def span_eval(y_true, y_pred, eval_='any_overlap'):

    # Check lengths
    assert len(y_true) == len(y_pred), "length error"
    for yt, yp in zip(y_true, y_pred):
        assert len(yt) == len(yp), "length mismatch"
    
    # Get spans
    spans_true = BIO_to_span_doc(y_true)
    spans_pred = BIO_to_span_doc(y_pred)

    # Output dictionary
    d =  OrderedDict()
    
    # True positive count
    TP = 0
    
    # Loop on sentences
    for T, P in zip(spans_true, spans_pred):

        # Loop on true spans
        for t in T:

            # Loop on predicted spans
            for p in P:
                
                # Match found
                if t.compare(p, eval_):
                    TP += 1
                    break
    d['TP'] = TP

    # Count number of spans    
    cnt = lambda doc: len([True for sent in doc for span in sent])

    # Spans in prediction
    NP = cnt(spans_pred)
    d['NP'] = NP
   
    # Spans in truth
    NT = cnt(spans_true)
    d['NT'] = NT
  
    div = lambda num, den: 0 if den==0 else num/float(den)
  
    # Precision
    P = div(TP, NP)
    d['P'] = P
    
    # Recall
    R = div(TP, NT)
    d['R'] = R
                
    # F1
    F1 = div(2*P*R, P + R)
    d['F1'] = F1

    return d
'''            
       
def span_eval_multi(y_true, y_pred, BIO, eval_='any_overlap'):
        
    # Check sequence count    
    assert len(y_true) == len(y_pred), "length mismatch"
    
    # Convert to nested dictionaries
    y_true = list_to_nested_dict(y_true)
    y_pred = list_to_nested_dict(y_pred)
    
    # Loop over event-entity combinations and evaluate
    scores = []
    for event, entities in y_true.items():
        for entity, labels in entities.items():
            if BIO[entity]:
                
                d = OrderedDict()
                d['event'] = event
                d['entity'] = entity
                d.update(span_eval( \
                                        y_true = y_true[event][entity],
                                        y_pred = y_pred[event][entity], 
                                        eval_ = eval_))
                scores.append(d)
    
    # Return data frame
    return pd.DataFrame(scores)
    

class Scorer(object):
    
    def __init__(self,  \
                neg_label, 
                ):
        '''
        Scorer
        
        '''
        self.neg_label = neg_label       
        
    def preprocess(self, y_true, y_pred, y_prob):          
        '''
        Preprocess predictions
        '''        
        # Determine if list of list (sequence of sequence)
        seq_of_seq = isinstance(y_true[0], list)
        
        # Check lengths
        assert len(y_true) == len(y_pred), "length error"
        if seq_of_seq:
            for t, p in zip(y_true, y_pred):
                assert len(t) == len(p), "length error"        
                         
        # Flatten if necessary
        if seq_of_seq:
            y_true = flatten_(y_true)
            y_pred = flatten_(y_pred)
            if y_prob is not None:
                y_prob = flatten_(y_prob)
                  
        return (y_true, y_pred, y_prob)
    
    def fit(self, y_true, y_pred, y_prob=None, params=None):
        '''        
        Score predictions
        '''
        
        # Preprocess        
        y_true, y_pred, y_prob = self.preprocess(y_true, y_pred, y_prob)

        # Get scores
        df = perf_metrics(y_true, y_pred, y_prob, self.neg_label)
        
        # Include each parameter in data frame       
        if params is not None:
            df = add_params_to_df(df, params)
        
        return df



        

    


    
    


class MultitaskScorer(object):
    
    def __init__(self, neg_label, BIO,
                
        ):


        self.neg_label = neg_label
        self.BIO = BIO
        
   
    def preprocess(self, y_true, y_pred, y_prob=None):                    
        '''
        Preprocess labels
        '''


        # Check lengths        
        assert len(y_true) == len(y_pred), "length mismatch"

        # Get event-entity combos
        combos = None
        for y in y_pred:
            if y is not None:
                combos = {evt:list(ents.keys()) \
                                             for evt, ents in y.items()}
                break
        assert combos is not None, "no unmasked sentences"

        # Replace None with appropriate negative label(s)
        # Loop on sentences
        for i in range(len(y_true)):
            
            # Current sentence is None (i.e. was masked)
            if y_pred[i] is None:
                
                y = {}
                for evt, ents in combos.items():
                    y[evt] = {}
                    for ent in ents:
                        
                        # True labels for current sentence and
                        # event-entity combination
                        t = y_true[i][evt][ent]
                        
                        # Sequence tags
                        if isinstance(t, list):
                            y[evt][ent] = len(t)*[self.neg_label[ent]]
                            
                        # Sentence-level label
                        else:
                            y[evt][ent] = self.neg_label[ent]
        
                
                y_pred[i] = y

            
        y_true = list_to_nested_dict(y_true)
        y_true = flatten_multitask(y_true)
        
        y_pred = list_to_nested_dict(y_pred)
        y_pred = flatten_multitask(y_pred)
        
        
        # Remove BIO prefixes
        # Loop on sentences
        for event, entities in y_true.items():
            for entity, labels in entities.items():
                
                if self.BIO[entity]:
            
                    y_true[event][entity] = postprocess_labels_seq( \
                                    seq = y_true[event][entity], \
                                    BIO = self.BIO[entity], 
                                    pad_start = False,
                                    pad_end = False)

                    y_pred[event][entity] = postprocess_labels_seq( \
                                    seq = y_pred[event][entity], \
                                    BIO = self.BIO[entity], 
                                    pad_start = False,
                                    pad_end = False)                
        
        
        if y_prob is None:
            y_prob = {evt: {ent:None for ent, _ in ents.items()} \
                                        for evt, ents in y_pred.items()}
        else:
            y_prob = list_to_nested_dict(y_prob)
            y_prob = flatten_multitask(y_prob)

        return (y_true, y_pred, y_prob)


    def fit(self, y_true, y_pred, y_prob=None, params=None):
        '''        
        Score predictions
        '''
        
        # Preprocess labels
        y_true, y_pred, y_prob = self.preprocess(y_true, y_pred, y_prob)


        # Get precision, recall, f, and support
        dfs = []

        # Loop on label types
        y_flat = {evt: {} for evt, _ in y_pred.items()}
        for evt, ents in y_pred.items():
            for ent, labs in ents.items():
                
                # Precision, recall, and other metrics
                df = perf_metrics( \
                        y_true = y_true[evt][ent], 
                        y_pred = y_pred[evt][ent], 
                        y_prob = y_prob[evt][ent], 
                        neg_label = self.neg_label[ent])
           
                # Include entity in event
                df.insert(loc=0, column='entity', value=ent)
                df.insert(loc=0, column='event', value=evt)

                # Append current            
                dfs.append(df)
            
        # Merge data frames
        df = pd.concat(dfs)            

        # Include each parameter in data frame       
        if params is not None:
            df = add_params_to_df(df, params)

        return df


def add_params_to_df(df, params):

    # Loop on Level 1
    for p1, v1 in params.items():
        
        # Level 1 as dictionary
        if isinstance(v1, dict):
            
            # Loop on level 2
            for p2, v2 in v1.items():
                
                # Level 2 as dictionary
                if isinstance(v2, dict):
            
                    # Loop on level 3
                    for p3, v3 in v2.items():
                        
                        # Level 3 is dictionary
                        if isinstance(v3, dict):
                            df[str((p1, p2, p3))] = str(v3)
                        
                        # Level 3 is not dict, list, or array            
                        elif not isinstance(v3, (list, np.ndarray)):
                            df[str((p1, p2, p3))] = v3
                                            
                # Level 2 is not dict, list, or array            
                elif not isinstance(v2, (list, np.ndarray)):
                    df[str((p1, p2))] = v2
                    
        # Level 1 is not dict, list, or array                
        elif not isinstance(v1, (list, np.ndarray)):
            df[p1] = v1
        
        
        
    return df

def build_e2e(X_e2e, y_ss, mask, default_val):
    '''
    Build end-to-end from single Sage
    '''
    
    # Indices of mask keep (i.e. 1)
    mask_idx = np.where(np.array(mask))[0]
    
    # Check dimensions
    assert len(mask_idx) == len(y_ss), '''Length mismatch:
                            {} vs {}'''.format(len(mask_idx), len(y_ss))
    
    # Create initialized list with same dimensions as truth
    seq_of_seq = isinstance(y_ss[0], list)
    if seq_of_seq:
        y_e2e = [[default_val]*len(a) for a in X_e2e]
    # Labels are sequence
    else:
        y_e2e = [default_val]*len(X_e2e)        
        
         
    # Update values at mask locations
    for i, y_ in zip(mask_idx, y_ss):
        if seq_of_seq:
            msg = '''size mismatch: 
                y_e2e len = {}, y_ len == {}'''.format( \
                                            len(y_e2e[i]), len(y_))
            assert len(y_e2e[i]) == len(y_), msg
        y_e2e[i] = y_
    
    return y_e2e

'''
def get_complete_labels(y, y_pred, mask, neg_label):
    
    Reintegrate extracted (unmasked) labels
    
            
    # Create end-to-end probability
    y_pred_e2e = build_e2e( \
                    y = y, 
                    y_pred = y_pred, 
                    mask = mask, 
                    default_val = neg_label)
    
    return y_pred_e2e
'''

def get_complete_prob(y, y_prob, mask, neg_label):
    '''
    Reintegrate probabilities
    
    args:
        y_prob: list of dict of prob OR list of list of dict of prob
    '''

    # Determine if nested list
    seq_of_seq = isinstance(y[0], list)
    
    # Get first result
    if seq_of_seq:
        first_item = y_prob[0][0]
    else:
        first_item = y_prob[0]
    
    # Assign 100% probability mass to negative label
    default_val = {lab:float(lab==neg_label) \
                                for lab, _ in first_item.items()}
    
    # Create end-to-end probability
    y_prob_e2e = build_e2e( \
                    y = y, 
                    y_pred = y_prob, 
                    mask = mask, 
                    default_val = default_val)
    
    return y_prob_e2e
    
def get_masked(X, y, mask, seq_of_seq):
        
    # If no mask provide, include all
    if mask is None:
        mask = [1]*len(y)

            
    # Retain label shape
    if seq_of_seq:
        y_dims = [len(y_) for y_ in y]
    else:
        y_dims = len(y)
        
    # Use mask to extract subset of features and labels
    X_extract = apply_mask(X, mask)
    y_extract = apply_mask(y, mask)

    return (X_extract, y_extract, mask, y_dims)    

        
def flatten_seq(Y):
    return [y for yseq in Y for y in yseq]

def get_confusion_matrix(y_true, y_pred):
    '''
    Confusion matrix
    '''
    # Get confusion matrix and counts
    y_t = pd.Series(y_true, name='True')
    y_p = pd.Series(y_pred, name='Predicted')
    df = pd.crosstab(y_t, y_p, margins=True)
    return df

def perf_counts(y_true, y_pred, neg_label=[0]):
    '''
    Get counts (true positive, false-negative, etc.) for positive label    
    '''

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for t, p in zip(y_true, y_pred): 
        
        # Is true label negative?
        is_neg = p in neg_label
        is_pos = not is_neg
        
        # Are true and predicted labels equal?
        is_equal = p == t
        
        # True positives
        if is_equal and is_pos:
           tp += 1
    
        # False positive
        elif not is_equal and is_pos:
           fp += 1
    
        # True negative
        elif is_equal and is_neg:
           tn += 1
        
        # False-negative
        elif not is_equal and is_neg:
           fn += 1

    return (tp, fp, tn, fn)

def pr_curve(y_true, y_prob, n=300):
    '''
    Precision-recall curve, interpolated at fixed points
    '''

    # Get precision recall curve
    p, r, t = precision_recall_curve(y_true, y_prob)

    # Define equally spaced recall points
    #R = np.linspace(0, 1, n)
    
    # Interpolate precision values at equally spaced recall points
    #P = np.interp(R, np.flip(r, 0), np.flip(p, 0))

    return (p, r)


def roc_curve_(y_true, y_prob, n=300):
    '''
    Receiver operator characteristic, interpolated at fixed points
    '''

    # Get precision recall curve
    fpr, tpr, t = roc_curve(y_true, y_prob)

    # Define equally spaced recall points
    #FPR = np.linspace(0, 1, n)

    # Interpolate precision values at equally spaced recall points
    #TPR = np.interp(FPR, fpr, tpr)
        
    return (fpr, tpr)

def optimum_f1(precision, recall):    
    
    # Make sure precision and recall are numpy arrays
    precision = np.array(precision)
    recall = np.array(recall)
    
    # Calculate f1
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.true_divide(2*precision*recall, precision + recall)
        f1[~ np.isfinite(f1)] = 0  # -inf inf NaN
    
    # Calculate Max f1
    f1_max = np.max(f1)
    idx_max = np.argmax(f1)
    
    # Calculate precision and recall at maximum f1
    p_max = precision[idx_max]
    r_max = recall[idx_max]
    
    return (p_max, r_max, f1_max)


def perf_metrics(y_true, y_pred, y_prob=None, neg_label=0):


    '''
    Based on code from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    '''

    # Get positive labels
    labels = list(set(y_true) | set(y_pred))
    pos_labels = labels[:]
    if neg_label in pos_labels:
        pos_labels.remove(neg_label)
    
    # If no probabilities provided, assume all positive probabilities 0
    if y_prob == None:
        y_prob = [{lab: float(lab==neg_label) \
                                         for lab in labels}]*len(y_true)
    '''
    Individual labels (as binary) and micro
    '''    
    # Performance cases
    cases = {lab:[lab] for lab in pos_labels}
    cases['micro'] = pos_labels
    
    scores = []

    # Loop on cases
    for name, labels in cases.items():
        
        # Indicator function for label and probabilities
        y_true_bin = [int(y==lab) for lab in labels for y in y_true]
        y_pred_bin = [int(y==lab) for lab in labels for y in y_pred]
        y_prob_bin = [y.get(lab, 0) for lab in labels for y in y_prob]

   
        # Precision, recall, and letter f1 scores, model default 
        # (i.e. majority vote)
        p, r, f1, _ = precision_recall_fscore_support( \
                  y_true_bin, y_pred_bin, pos_label=1, average='binary')
        
        # Counts, true positives, false negatives, etc.
        tp, fp, tn, fn = perf_counts(y_true_bin, y_pred_bin, neg_label=[0])

        scores.append(( name,  tp, fp, tn, fn, p, r, f1))
        columns =      [LABEL, TP, FP, TN, FN, P, R, F1]
    
    df = pd.DataFrame(scores, columns=columns)

    return df

def perf_metrics_dict(y_true, y_pred, neg_label):
    
    dfs = []
    for evt_typ in y_pred:
 
        # Precision, recall, and other metrics
        df = perf_metrics( \
                y_true = y_true[evt_typ], 
                y_pred = y_pred[evt_typ], 
                neg_label = neg_label[evt_typ])
   
        # Include entity in event
        df.insert(loc=0, column='event type', value=evt_typ)

        # Append current            
        dfs.append(df)
        
    # Merge data frames
    df = pd.concat(dfs)   

    return df

def flatten_multitask(y):
    '''
    Iterate through dictionary of labels and flatten 
    if sequence of sequences
    '''
    
    # Loop on label types
    y_flat = {evt: {} for evt, _ in y.items()}
    for evt, ents in y.items():
        for ent, labs in ents.items():
        
            if labs is not None:
        
                # Flatten, if sequence of sequences
                if isinstance(labs[0], list):
                    labs = flatten_(labs)
            
            y_flat[evt][ent] = labs

    return y_flat
    

def get_scores_df(scores_ss, columns=None, eval_range=None):
        
    # Single state results
    df = pd.DataFrame.from_dict(scores_ss, orient='index')

    # Truncate columns
    if columns is not None:
        df = df[columns] 
            
    if eval_range is not None:
        df['eval_range'] = eval_range
            
    # Make indices a column
    df.reset_index(inplace=True)
    columns = df.columns.tolist()
    columns[0] = LABEL
    df.columns = columns

    return df


