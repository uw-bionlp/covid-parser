


import json
import os
import joblib
import re    
import shutil
import pandas as pd

import numpy as np
import copy
import torch

from corpus.event2 import EventScorer as Scorer
from utils.misc import flatten
from utils.misc import dict_to_list, list_to_dict
from utils.misc import nested_dict_to_list, list_to_nested_dict
from utils.seq_prep import preprocess_labels_seq, postprocess_labels_seq
from utils.seq_prep import preprocess_tokens_doc, postprocess_tokens_doc
from utils.scoring import build_e2e

from models.Base import Base


from models.event_extractor.model import EventExtractor as Estimator

from constants import *




class EventExtractorWrapper(Base):
 
    '''
    Sequence classifier
    '''
    
    def __init__(self, 
        hyperparams = None, 
        metric = None, 
        average = None, 
        path = None, 
        descrip = 'Multitask', 
        ):


        self.BIO = True
        self.neg_label = OUTSIDE
        self.pad_start = True
        self.pad_end = True 
                       
        scorer = Scorer(by_doc = True)
                
        super().__init__( \
            estimator_class = Estimator, 
            fit_method = None, 
            prob_method = None, 
            pred_method = None, 
            get_params_method = 'get_params', 
            hyperparams = hyperparams, 
            metric = metric, 
            average = average, 
            scorer = scorer, 
            neg_label = None, 
            path = path, 
            feat_params = None, 
            event = 'Multi-task',
            entity = 'Multi-task', 
            model_type = 'Multi-task',
            descrip = 'Event extractor', 
            ) 

    #@Override
    def predict_sub_op(self, X, y=None, pass_true=False, **kwargs):
        '''
        Predict labels, without any masking
        '''
        
        # Preprocess X (y not available)
        X, y = self.preprocess(X, y)
        
        # Get predictions
        if (y is not None) and pass_true:
            y = self.estimator.predict(X, y, **kwargs)
        else:
            y = self.estimator.predict(X, **kwargs)
        
        # Postprocess predictions
        _, y = self.postprocess(None, y)
        
        return y 

    #@Override
    def score(self, X, y, path=None, pass_true=False, **kwargs):
        '''
        Predict labels and score result
        '''
               
        # Get predictions
        if (y is not None) and pass_true:
            y_pred = self.predict_sub_op(X, y=y, pass_true=pass_true, **kwargs)
        else:
            y_pred = self.predict_sub_op(X, **kwargs)

        # Preprocess labels
        _, y = self.preprocess(None, y)
                       
        # Score single state results
        scores = self.scorer.fit(y, y_pred, params=self.get_params())
                                               
        # Save results
        if path is not None:
            self.save_results(path, y_pred, scores)
            
        return (y_pred, scores)

    