import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.nn import util
from torch.nn.utils.clip_grad import clip_grad_norm_

from pytorch_memlab import profile
from collections import OrderedDict
import math
from tensorboardX import SummaryWriter
import re

import os
import errno
from datetime import datetime
from tqdm import tqdm
import numpy as np
import logging
import joblib
import math
from functools import partial

from models.crf import MultitaskCRF, MultitaskSpanExtractor, seq_tags_to_spans, multitask_seq_tags_to_spans
from models.attention import MultitaskAttention, MultiHeadAttention
from models.utils import create_mask, tensor_summary, create_Tensorboard_writer, batched_select, one_hot, get_predictions, get_device
from models.recurrent import Recurrent
from models.event_extractor.dataset import EventExtractorDataset, map_1D, export_idx, get_num_tags, span_maps, seq_maps

from models.training import MultiTaskLoss, reduce_loss, get_loss
from models.span_embedder import SpanEmbedder, span_embed_agg
from models.span_scoring import SpanScorerGold, SpanScorerCRF, num_keep_from_one_hot, SpanScorerFFNN, span_pruner
from models.argument_scoring import ArgumentScorerRule, ArgumentScorerGold, ArgumentScorerLearned, prune_arguments, distance_features


from utils.misc import nested_dict_to_list, list_to_nested_dict
from utils.timer import Timers


from constants import *


def to_device(X, Y, mask, device):

    X = X.to(device)   
    
    if Y is not None:
        for outer, y in Y.items():
            
            if isinstance(y, dict):
                for inner, v in y.items():                
                    Y[outer][inner] = Y[outer][inner].to(device)
            else:            
                Y[outer] = Y[outer].to(device)
    mask = mask.to(device)

    return (X, Y, mask)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx


def multitask_loss(loss, entity):
    loss = [entities[entity] for _, entities in loss.items()]
    loss = torch.stack(loss).mean()
    return loss
            

def sent_to_seq_feat(X, seq_len):
    '''
    Convert sentence-level features to sequence features
    '''   
    return X.unsqueeze(1).repeat(1, seq_len, 1)

def dict_sent_to_seq_feat(X, seq_len):
    '''
    Convert multiple sentence-level features to a single set of 
    sequence features
    '''  
    
    # Get sentence-level features and concatenate
    X = torch.cat([x for k, x in X.items()], 1)
    
    # Convert to sequence features
    X = sent_to_seq_feat(X, seq_len)
    
    return X


def max_grad(parameters):
    '''
    Get maximum gradient
    '''   
        
    # Get parameters with gradient
    param_with_grad = [p for p in parameters if p.grad is not None]
        
    # Calculate maximum gradient across all parameters
    max_grad =  max(p.grad.data.abs().max() for p in param_with_grad)
    
    return max_grad

def merge_entities(a, b, entity):
    
    for event, val in b.items():
        if event not in a.keys():
            a[event] = {}
        a[event][entity] = val
    return a
    

def to_arg_name(name):
    return '{}_arg'.format(name)
def from_arg_name(name):
    return re.sub('_arg', '', name)


class EventExtractor(nn.Module):

    def __init__(self, \
    
        # Labels
        label_def, 
        source = None,
        max_span_width = 8, 
        min_span_width = 1,   
        
        span_scoring = SPAN_CRF,
        argument_scoring = ARGUMENT_RULE, 

        # Word embeddings
        X_includes_mask = False,
        use_xfmr = False, 
        xfmr_type = None, 
        xfmr_dir = None,
        word_embed_source = None, 
        word_embed_path = None,
       
        
        # Recurrent layer
        rnn_type = 'lstm',
        rnn_input_size = 768, 
        rnn_hidden_size = 20, 
        rnn_num_layers = 1,
        rnn_bias = True,
        rnn_input_dropout = 0.0, 
        rnn_layer_dropout = 0.0,
        rnn_output_dropout = 0.0,
        rnn_bidirectional = True,
        rnn_stateful = False,
        rnn_layer_norm = True,
        
        #---------------------------------------------------------------
        # Span embedding
        #---------------------------------------------------------------
        # Endpoint extractor parameters        
        span_embed_use_endpoint = True, 
        span_embed_combination = "x,y,x*y",
        span_embed_width_embedding_dim = 20,

        # Attention parameters      
        span_embed_use_attentive = True, 
        #span_embed_use_local_weights = True,
        span_embed_single_embedder = False,
       
        # Heuristic position parameters
        span_embed_use_position_heuristic = False,
        span_embed_position_hidden_size = 30,
        span_embed_min_timescale = 1.0,
        span_embed_max_timescale = 1.0e4,
        
        # Learned of position parameters
        span_embed_use_position_learned = False,
        
        # FFNN projection parameters
        span_embed_project = False,
        span_embed_hidden_size = 50,
        span_embed_activation = 'tanh',
        span_embed_dropout = 0.0,
        
        
        #---------------------------------------------------------------
        # Span scoring
        #---------------------------------------------------------------
        span_scorer_hidden_size = 100, 
        span_scorer_activation = 'relu',
        span_scorer_dropout = 0.0,
        span_scorer_attn_feat = False,        
        span_scorer_trig_attn_based = False,
        span_scorer_supervised_attn = False,
        limit_crf_decode = False,
        # span_scorer_seq_feat = False, 


        #---------------------------------------------------------------
        # Span pruning
        #---------------------------------------------------------------

        span_prune_spans_per_word = 50,
        span_prune_agg_type = 'max',
        
        prune_overlapping = False,
               
        # Argument scoring
        arg_hidden_size = 100, 
        arg_dropout = 0.20,
        arg_use_label_scores = True,
        arg_use_dist_features = True, 
        arg_activation = 'relu',
                
        # Logging
        log_dir = None,
        log_subfolder = True,
        
        # Training
        max_len = 50,
        num_epochs = 100,
        batch_size = 50,
        num_workers = 6,
        learning_rate = 0.005,
        grad_max_norm = 1,
        loss_reduction = 'sum',
       
        
        ):
        super(EventExtractor, self).__init__()


        # Labels
        self.label_def = label_def
        self.source = source
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        
        # Word embeddings     
        self.X_includes_mask = X_includes_mask
        self.use_xfmr = use_xfmr  
        self.xfmr_type = xfmr_type
        self.xfmr_dir = xfmr_dir
        self.word_embed_source = word_embed_source
        self.word_embed_path = word_embed_path
       
        
        # Recurrent layer
        self.rnn_type = rnn_type
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bias = rnn_bias
        self.rnn_batch_first = True 
        self.rnn_input_dropout = rnn_input_dropout
        self.rnn_layer_dropout = rnn_layer_dropout
        self.rnn_output_dropout = rnn_output_dropout
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_stateful = rnn_stateful
        self.rnn_layer_norm = rnn_layer_norm
        
        #---------------------------------------------------------------
        # Span embedding
        #---------------------------------------------------------------
        # Endpoint extractor parameters        
        self.span_embed_use_endpoint = span_embed_use_endpoint
        self.span_embed_combination = span_embed_combination
        self.span_embed_width_embedding_dim = span_embed_width_embedding_dim
        
        # Attention parameters            
        self.span_embed_use_attentive = span_embed_use_attentive
        #self.span_embed_use_local_weights = span_embed_use_local_weights
        
        # Heuristic position parameters
        self.span_embed_use_position_heuristic = span_embed_use_position_heuristic
        self.span_embed_position_hidden_size = span_embed_position_hidden_size
        self.span_embed_min_timescale = span_embed_min_timescale
        self.span_embed_max_timescale = span_embed_max_timescale
               
        # Learned of position parameters
        self.span_embed_use_position_learned = span_embed_use_position_learned
        
        # FFNN projection parameters
        self.span_embed_project = span_embed_project
        self.span_embed_hidden_size = span_embed_hidden_size
        self.span_embed_activation = span_embed_activation
        self.span_embed_dropout = span_embed_dropout
        self.span_embed_single_embedder = span_embed_single_embedder
        
        
        # Span scoring
        self.span_scorer_hidden_size = span_scorer_hidden_size
        self.span_scorer_activation = span_scorer_activation
        self.span_scorer_dropout = span_scorer_dropout
        
        self.span_scorer_attn_feat = span_scorer_attn_feat # Use multi head attention features with span embedding
        self.span_scorer_trig_attn_based = span_scorer_trig_attn_based
        self.span_scorer_supervised_attn = span_scorer_supervised_attn
        self.limit_crf_decode = limit_crf_decode
        
        #self.span_scorer_seq_feat = span_scorer_seq_feat
        
        # Span pruning
        self.span_prune_spans_per_word = span_prune_spans_per_word
        self.span_prune_agg_type = span_prune_agg_type
        
        self.prune_overlapping = prune_overlapping
        
        # Argument scoring
        self.arg_hidden_size = arg_hidden_size
        self.arg_dropout = arg_dropout
        self.arg_use_label_scores = arg_use_label_scores
        self.arg_use_dist_features = arg_use_dist_features
        self.arg_activation = arg_activation
               
        # CRF
        
        # Logging
        self.log_dir = log_dir
        self.log_subfolder = log_subfolder
        self.timers = Timers(self.log_dir)
        
        # Training
        self.max_len = max_len
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.grad_max_norm = grad_max_norm
        self.loss_reduction = loss_reduction
               
        self.span_scoring = span_scoring
        self.argument_scoring = argument_scoring

        # Get number of tags        
        self.num_tags = get_num_tags(label_def)


        self.label_to_id, self.id_to_label = span_maps(label_def)        
        self.seq_tag_to_id, self.id_to_seq_tag, self.tag_to_span_fn,  self.num_tags_seq, self.seq_tag_constraints \
                                = seq_maps(self.id_to_label)

        '''
        Input layer
        '''
             
        # Define recurrent layer
        self.rnn = Recurrent( \
            input_size = self.rnn_input_size, 
            output_size = self.rnn_hidden_size, 
            type_ = self.rnn_type,
            num_layers = self.rnn_num_layers, 
            bias = self.rnn_bias, 
            batch_first = self.rnn_batch_first, 
            bidirectional = self.rnn_bidirectional,
            stateful = self.rnn_stateful,    
            dropout_input = self.rnn_input_dropout, 
            dropout_rnn = self.rnn_layer_dropout,
            dropout_output = self.rnn_output_dropout,
            layer_norm = self.rnn_layer_norm,
            )


        '''
        Span embedding
        '''
        
        if self.span_embed_single_embedder:
                self.embedder = SpanEmbedder( \
                    input_dim = self.rnn.output_size, 
                    
                    # Endpoint extractor parameters        
                    use_endpoint = self.span_embed_use_endpoint, 
                    endpoint_combos = self.span_embed_combination,
                    num_width_embeddings = self.max_span_width, 
                    span_width_embedding_dim = self.span_embed_width_embedding_dim,
                    
                    # Attention parameters    
                    use_attention = self.span_embed_use_attentive,
                    #use_local_weights = self.span_embed_use_local_weights,
                    
                    # Heuristic position parameters
                    use_position_heuristic = self.span_embed_use_position_heuristic,
                    max_seq_len = self.max_len,
                    position_hidden_size = self.span_embed_position_hidden_size,           
                    min_timescale = self.span_embed_min_timescale, 
                    max_timescale = self.span_embed_max_timescale,                 
                    
                    # Learned of position parameters
                    use_position_learned = self.span_embed_use_position_learned,
                    
                    # FFNN projection parameters
                    project = self.span_embed_project,
                    hidden_dim = self.span_embed_hidden_size,
                    activation = self.span_embed_activation,
                    dropout = self.span_embed_dropout,
                
                    # General config
                    span_end_is_exclusive = True)        
        
        else:
            self.embedder = nn.ModuleDict(OrderedDict())
            for k, lab_def in self.label_def.items():
                self.embedder[k] = SpanEmbedder( \
                    input_dim = self.rnn.output_size, 
                    
                    # Endpoint extractor parameters        
                    use_endpoint = self.span_embed_use_endpoint, 
                    endpoint_combos = self.span_embed_combination,
                    num_width_embeddings = self.max_span_width, 
                    span_width_embedding_dim = self.span_embed_width_embedding_dim,
                    
                    # Attention parameters    
                    use_attention = self.span_embed_use_attentive,
                    #use_local_weights = self.span_embed_use_local_weights,
                    
                    # Heuristic position parameters
                    use_position_heuristic = self.span_embed_use_position_heuristic,
                    max_seq_len = self.max_len,
                    position_hidden_size = self.span_embed_position_hidden_size,           
                    min_timescale = self.span_embed_min_timescale, 
                    max_timescale = self.span_embed_max_timescale,                 
                    
                    # Learned of position parameters
                    use_position_learned = self.span_embed_use_position_learned,
                    
                    # FFNN projection parameters
                    project = self.span_embed_project,
                    hidden_dim = self.span_embed_hidden_size,
                    activation = self.span_embed_activation,
                    dropout = self.span_embed_dropout,
                
                    # General config
                    span_end_is_exclusive = True)

        
        '''
        Span scoring
        '''


        self.span_scorer = nn.ModuleDict(OrderedDict())
        self.attn = nn.ModuleDict(OrderedDict())
        for k, lab_def in self.label_def.items():

            if self.span_scorer_attn_feat or \
               ((k == TRIGGER) and self.span_scorer_trig_attn_based):
                 self.attn[k] = MultiHeadAttention( \
                            input_dim = self.rnn.output_size, 
                            num_tags = self.num_tags[k], 
                            output_dim = 2, 
                            reduction = self.loss_reduction,
                            dropout = self.span_scorer_dropout,
                            use_supervised_attn = self.span_scorer_supervised_attn,
                            path = os.path.join(self.log_dir, 'alphas_{}.csv'.format(k)))


            if self.span_scorer_attn_feat:
                 seq_feat_size = self.attn[k].span_feat_size
            else:
                 seq_feat_size = 0
                
            # Gold span scoring
            if self.span_scoring == SPAN_GOLD:
                self.span_scorer[k] = SpanScorerGold( \
                                        num_tags = self.num_tags[k])

            # CRF span scoring
            elif self.span_scoring == SPAN_CRF:

                self.span_scorer[k] = SpanScorerCRF( \
                            embed_size = self.rnn.output_size,
                            tag_to_span_fn = self.tag_to_span_fn[k], 
                            num_tags_seq = self.num_tags_seq[k],
                            num_tags_span = self.num_tags[k],
                            constraints = None, # self.seq_tag_constraints[k],
                            incl_start_end = True, 
                            name = k,
                            timers = self.timers,
                            limit_decode = self.limit_crf_decode)
                                
            # Enum-prune span scoring
            elif self.span_scoring == SPAN_FFNN:
                
                if self.span_embed_single_embedder:
                    input_dim = self.embedder.output_dim + seq_feat_size
                else:
                    input_dim = self.embedder[k].output_dim + seq_feat_size
                self.span_scorer[k] = SpanScorerFFNN( \
                            input_dim = input_dim,
                            hidden_dim = self.span_scorer_hidden_size, 
                            num_tags = self.num_tags[k], 
                            activation = self.span_scorer_activation,
                            dropout = self.span_scorer_dropout,
                            loss_reduction = self.loss_reduction,
                            seq_feat_size = None)
                            #seq_feat_size = seq_feat_size)



                
        
        '''
        Argument scoring
        '''

        self.arg_scorer = nn.ModuleDict(OrderedDict())
        for k in list(self.label_def.keys())[1:]:
                    
            # Gold argument scoring
            if self.argument_scoring == ARGUMENT_GOLD:
                self.arg_scorer[k] = ArgumentScorerGold()

            # Rule-based argument scoring
            elif self.argument_scoring == ARGUMENT_RULE:
                self.arg_scorer[k] = ArgumentScorerRule( \
                                        mode = STATUS)
                
            # Learned argument scoring
            elif self.argument_scoring == ARGUMENT_LEARNED:

                
                
                input_dim = 0
                if self.span_embed_single_embedder:
                    input_dim += self.embedder.output_dim*2
                else:
                    input_dim += self.embedder[k].output_dim*2


                if self.arg_use_dist_features:
                    input_dim += 6
                if self.arg_use_label_scores:
                    input_dim += self.num_tags[TRIGGER]
                    input_dim += self.num_tags[k]
                    
                self.arg_scorer[k] = ArgumentScorerLearned( \
                        input_dim = input_dim, 
                        hidden_dim = self.arg_hidden_size, 
                        activation = self.arg_activation,
                        dropout = self.arg_dropout)
    #@profile                       
    def forward(self, X, y, mask=None, decode_=True, span_map=None, verbose=False):
        self.timers.start('forward')   
        
        '''
        Recurrent layer with normalization
        '''
        # Recurrent layer     
        self.timers.start('rnn')   
        H = self.rnn(X, mask)
        self.timers.stop('rnn')   

        '''
        Span embedding
        '''
        self.timers.start('embedder')  
        if self.span_embed_single_embedder:
            span_embed = self.embedder( \
                            sequence_tensor = H,
                            sequence_mask = mask, 
                            span_indices = y['span_indices'],
                            span_mask = y['span_mask'],
                            verbose = verbose)
        else:
            span_embed = OrderedDict()
            for k, embedder in self.embedder.items():
                span_embed[k] = embedder( \
                                sequence_tensor = H,
                                sequence_mask = mask, 
                                span_indices = y['span_indices'],
                                span_mask = y['span_mask'],
                                verbose = verbose)
        self.timers.stop('embedder')   

        '''
        Span scoring
        '''
        
        # Iterate over span types
        ind_alphas = OrderedDict()
        ind_span_scores = OrderedDict()
        ind_loss = OrderedDict()
        span_logits = OrderedDict()
        span_loss = OrderedDict()
        num_keep = OrderedDict()
        self.timers.start('span scorer')   
        for k, span_scorer in self.span_scorer.items():

            num_keep[k] = None

            if self.span_scorer_attn_feat:
                embed_tmp, ind_alphas[k], ind_loss[k] = self.attn[k].span_feat( \
                            seq_embed = H, 
                            seq_mask = mask, 
                            seq_labels = y['indicator_labels'][k], 
                            span_indices = y['span_indices'], 
                            seq_weights = y['indicator_weights'][k], 
                            span_embed = span_embed if self.span_embed_single_embedder else span_embed[k], 
                            verbose = verbose)

            else:
                embed_tmp = span_embed if self.span_embed_single_embedder else span_embed[k]
            
            # Use gold span labels    
            if self.span_scoring == SPAN_GOLD:            
                span_logits[k] = span_scorer(labels = y['mention_labels'][k])                         
                num_keep[k] = num_keep_from_one_hot(X)
                
            # Predict span labels using CRF
            elif self.span_scoring == SPAN_CRF:  
                span_logits[k],  span_loss[k] = span_scorer( \
                                        seq_tensor = H,
                                        seq_mask = mask,
                                        span_map = span_map,
                                        span_indices = y['span_indices'],
                                        seq_labels = y['seq_labels'][k]
                                        )
                num_keep[k] = num_keep_from_one_hot(X)
            
            # Predict span labels through enumeration and pruning
            elif self.span_scoring == SPAN_FFNN:

                if (k == TRIGGER) and self.span_scorer_trig_attn_based:
                    span_logits[k], ind_alphas[k], span_loss[k] = self.attn[k].span_logits( \
                                seq_embed = H, 
                                seq_mask = mask, 
                                seq_labels = y['indicator_labels'][k], 
                                span_indices = y['span_indices'], 
                                span_labels = y['mention_labels'][k],
                                seq_weights = y['indicator_weights'][k], 
                                verbose = verbose)
                else:
                    span_logits[k],  span_loss[k] = span_scorer( \
                                    embed = embed_tmp,
                                    mask = y['span_mask'],
                                    labels = y['mention_labels'][k],
                                    seq_feat = None,
                                    verbose = verbose)
                                                      
            else:
                raise ValueError('''Invalid span score type: 
                                        {}'''.format(self.span_scoring))

        self.timers.stop('span scorer')   
        
        '''
        Span pruning
        '''
        
        # Iterate over span types
        top_k_idx = OrderedDict()
        span_scores_top = OrderedDict()
        span_mask_top = OrderedDict()
        span_embed_top = OrderedDict()
        span_logits_top = OrderedDict()        
        span_indices_top = OrderedDict()
        self.timers.start('span pruner')   
        for k in self.span_scorer:

            # Prune spans, based on predicted labels scores
            top_k_idx[k], span_scores_top[k], span_mask_top[k] \
                     = span_pruner( \
                            logits = span_logits[k], 
                            mask = y['span_mask'],
                            seq_length = y['seq_length'], 
                            spans_per_word = self.span_prune_spans_per_word,
                            num_keep = num_keep[k])

            
            # Get top embeddings and logits
            se = span_embed if self.span_embed_single_embedder else span_embed[k]
            
            span_embed_top[k], span_logits_top[k], span_indices_top[k] = batched_select( \
                tensors = (se, span_logits[k], y['span_indices']), 
                indices = top_k_idx[k])
        self.timers.stop('span pruner')   

        '''
        Argument pruning
        '''

        arg_labels_top = OrderedDict()
        arg_mask_top = OrderedDict()  
        self.timers.start('argument pruner')         
        for k in self.arg_scorer:   
            arg_labels_top[k], arg_mask_top[k] = prune_arguments( \
                                arg_labels = y['arg_labels'][k], 
                                trig_indices_top = top_k_idx[TRIGGER], 
                                arg_indices_top  = top_k_idx[k],
                                trig_mask_top = span_mask_top[TRIGGER], 
                                arg_mask_top  = span_mask_top[k])
        self.timers.stop('argument pruner')

        '''
        Argument scoring
        '''

        # Use gold argument labels
        arg_logits_top = OrderedDict()
        arg_loss = OrderedDict()
        self.timers.start('argument scorer')   
        for k, arg_scorer in self.arg_scorer.items():   

            # Gold arguments
            if self.argument_scoring == ARGUMENT_GOLD:
                arg_logits_top[k] = arg_scorer( \
                                arg_labels = arg_labels_top[k])

        
            # Rule-based (distance-based) arguments
            elif self.argument_scoring == ARGUMENT_RULE:
                arg_logits_top[k] = arg_scorer( \
                                trig_scores = span_logits_top[TRIGGER],
                                trig_spans = span_indices_top[TRIGGER],
                                trig_mask = span_mask_top[TRIGGER],
                                entity_scores = span_logits_top[k], 
                                entity_spans = span_indices_top[k],
                                entity_mask = span_mask_top[k])
                               
            # Predict argument assignments
            elif self.argument_scoring == ARGUMENT_LEARNED:


                if self.arg_use_dist_features:
                    dist_feat = distance_features(
                                trig_scores = span_logits_top[TRIGGER], 
                                trig_spans = span_indices_top[TRIGGER], 
                                trig_mask = span_mask_top[TRIGGER],
                                entity_scores = span_logits_top[k], 
                                entity_spans = span_indices_top[k], 
                                entity_mask = span_mask_top[k],
                                feature_type = None)
                else:
                    dist_feat = None
                
                # Trigger and argument embeddings     
                trig_embed = span_embed_top[TRIGGER]
                arg_embed  = span_embed_top[k]
                
                # Include logits with embeddings
                if self.arg_use_label_scores:
                    trig_embed = torch.cat((trig_embed, span_logits_top[TRIGGER]), dim=2)
                    arg_embed =  torch.cat((arg_embed, span_logits_top[k]), dim=2)
           
                
                arg_logits_top[k] = arg_scorer( \
                                        trig_embed = trig_embed,
                                        arg_embed  = arg_embed,
                                        additional_feat = dist_feat)
                                                
                arg_loss[k] = get_loss( \
                                            scores = arg_logits_top[k], 
                                            labels = arg_labels_top[k], 
                                            mask = arg_mask_top[k], 
                                            reduction = self.loss_reduction)                                    

            else:
                raise ValueError("Invalid argument ID: {}".format( \
                                                          self.argument_scoring))
        self.timers.stop('argument scorer')

        '''
        Gather results
        '''
        y_out = {}
        y_out['mention_scores'] = span_logits_top
        y_out['mention_spans'] = span_indices_top
        y_out['span_mask'] = span_mask_top
        y_out['arg_scores'] = arg_logits_top
        y_out['arg_mask'] = arg_mask_top
        y_out['alphas'] = ind_alphas
        
        loss = OrderedDict()
        if self.span_scorer_attn_feat:
            loss['ind'] = ind_loss
        loss['span'] = span_loss
        loss['arg'] = arg_loss
    
        self.timers.stop('forward')    
        
        return (loss, y_out)

    def predict_fast(self, X, embedding_model, tokenizer_model, device=None, **kwargs):

        # Create data set
        dataset = EventExtractorDataset( \
                                X = X, 
                                Y = None, 
                                max_len = self.max_len, 
                                label_def = self.label_def,
                                use_xfmr = self.use_xfmr,
                                xfmr_type = self.xfmr_type,
                                xfmr_dir = self.xfmr_dir,
                                word_embed = self.word_embed_path, 
                                max_span_width = self.max_span_width, 
                                min_span_width = self.min_span_width,
                                num_workers = self.num_workers,
                                device = device,
                                X_includes_mask = self.X_includes_mask,
                                tokenizer_model = tokenizer_model,
                                embedding_model = embedding_model)
        
        # Create data loader
        dataloader = data_utils.DataLoader(dataset, \
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        # Set number of cores
        torch.set_num_threads(self.num_workers)

        # Loop on mini-batches
        events_by_sent = []
        spans_by_sent = []
        alphas = []
        for i, (X_bat, y_bat, mask_bat) in enumerate(dataloader):

            X_bat, y_bat, mask_bat = to_device(X_bat, y_bat, mask_bat, device)

            # Push data through model
            _, y_pred_bat = self( \
                            X = X_bat, 
                            y = y_bat, 
                            mask = mask_bat,
                            decode_ = True,
                            span_map = dataset.span_map,
                            verbose = i == 0)          

            # Decode tensor span results
            spans_by_sent.extend(dataset.decode_spans( \
                        scores = y_pred_bat['mention_scores'], 
                        spans = y_pred_bat['mention_spans'],
                        mask = y_pred_bat['span_mask'],
                        prune_overlapping = self.prune_overlapping,
                        verbose = i == 0))
            
            # Decode tensor event results
            events_by_sent.extend(dataset.decode_events( \
                        mention_scores = y_pred_bat['mention_scores'], 
                        mention_spans = y_pred_bat['mention_spans'],
                        mention_mask = y_pred_bat['span_mask'],
                        arg_scores = y_pred_bat['arg_scores'],
                        arg_mask = y_pred_bat['arg_mask'],
                        prune_overlapping = self.prune_overlapping,
                        verbose = i == 0))

        spans_by_doc = dataset.by_doc(spans_by_sent)
        events_by_doc = dataset.by_doc(events_by_sent)
        
        return events_by_doc

    def predict(self, X, y=None, device=None, **kwargs):
        '''
        Train multitask model
        '''

        device = get_device(device)
        self.to(device)

        # Configure training mode
        self.eval()

        # Print model summary
        self.get_summary()

        # Create data set
        dataset = EventExtractorDataset( \
                                X = X, 
                                Y = y, 
                                max_len = self.max_len, 
                                label_def = self.label_def,
                                use_xfmr = self.use_xfmr,
                                xfmr_type = self.xfmr_type,
                                xfmr_dir = self.xfmr_dir,
                                word_embed = self.word_embed_path, 
                                max_span_width = self.max_span_width, 
                                min_span_width = self.min_span_width,
                                num_workers = self.num_workers,
                                device = device,
                                X_includes_mask = self.X_includes_mask)
        
        # Create data loader
        dataloader = data_utils.DataLoader(dataset, \
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        # Set number of cores
        torch.set_num_threads(self.num_workers)

        # Loop on mini-batches
        events_by_sent = []
        spans_by_sent = []
        alphas = []
        for i, (X_bat, y_bat, mask_bat) in enumerate(dataloader):

            X_bat, y_bat, mask_bat = to_device(X_bat, y_bat, mask_bat, device)

            # Push data through model
            _, y_pred_bat = self( \
                            X = X_bat, 
                            y = y_bat, 
                            mask = mask_bat,
                            decode_ = True,
                            span_map = dataset.span_map,
                            verbose = i == 0)          

            # Decode tensor span results
            spans_by_sent.extend(dataset.decode_spans( \
                        scores = y_pred_bat['mention_scores'], 
                        spans = y_pred_bat['mention_spans'],
                        mask = y_pred_bat['span_mask'],
                        prune_overlapping = self.prune_overlapping,
                        verbose = i == 0))
            
            # Decode tensor event results
            events_by_sent.extend(dataset.decode_events( \
                        mention_scores = y_pred_bat['mention_scores'], 
                        mention_spans = y_pred_bat['mention_spans'],
                        mention_mask = y_pred_bat['span_mask'],
                        arg_scores = y_pred_bat['arg_scores'],
                        arg_mask = y_pred_bat['arg_mask'],
                        prune_overlapping = self.prune_overlapping,
                        verbose = i == 0))

        spans_by_doc = dataset.by_doc(spans_by_sent)
        events_by_doc = dataset.by_doc(events_by_sent)
        
        return events_by_doc
        

    def get_summary(self):
        '''
        Generate and print summary of model
        '''        
        
        # Print model summary
        logging.info("\n")
        logging.info("Model summary")
        logging.info(self)
        
        # Print trainable parameters
        logging.info("\n")
        logging.info("Trainable parameters")
        for k, param in self.named_parameters():
            if param.requires_grad:
                logging.info('\t{}\t{}'.format(k, param.size()))
        
        logging.info("\n")
        num_p = sum(p.numel() for p in self.parameters() \
                                                     if p.requires_grad)
        num_pM = num_p/1e6
        logging.info("Total trainable parameters:\t{:.1f} M".format(num_pM))
        logging.info("\n")



