

import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.enabled = False
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from pytorch_models.recurrent import Recurrent

from tensorboardX import SummaryWriter

import pandas as pd
import os
import errno
from datetime import datetime
import numpy as np
import logging
from tqdm import tqdm
import joblib
import math
from collections import OrderedDict


from models.word2vec import get_w2v_embed
from pytorch_models.crf import MultitaskCRF
from pytorch_models.attention import MultitaskAttention
from pytorch_models.multitask.dataset import MultitaskDataset
from pytorch_models.utils import create_Tensorboard_writer
from utils.misc import nested_dict_to_list, list_to_nested_dict, list_to_dict, dict_to_list
from pytorch_models.training import MultiTaskLoss
from pytorch_models.utils import get_device, mem_size
from pytorch_models.multitask.dataset import get_label_map
from pytorch_models.utils import loss_reduction

from constants import *


class MultitaskEstimator(nn.Module):
    '''
    
    
    args:
        rnn_type: 'lstm'
        rnn_input_size: The number of expected features in the input x
        rnn_hidden_size: The number of features in the hidden state h
        rnn_num_layers: Number of recurrent layers
        rnn_bias: If False, then the layer does not use bias weights b_ih and b_hh.
        rnn_batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        rnn_dropout = If non-zero, introduces a Dropout layer on the 
                      outputs of each LSTM layer except the last layer, 
                      with dropout probability equal to dropout.
        rnn_bidirectional: If True, becomes a bidirectional LSTM. 
        
        
        
        
        
        
        
        
        
        https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    '''



    def __init__(self, \
    
        # Labels
        source,
        event_types,
        label_def, 
        
        # Word embeddings
        word_embed_source = None, 
        word_embed_dir = None,

        xfmr_name = None,
        xfmr_type = None,
        xfmr_dir = None,
        use_xfmr = False,
            
        # Recurrent layer
        rnn_type = 'lstm',
        rnn_input_size = 768, 
        rnn_hidden_size = 100, 
        rnn_num_layers = 1,
        rnn_bias = True,
        rnn_input_dropout = 0.0,
        rnn_layer_dropout = 0.0,
        rnn_output_dropout = 0.0,
        rnn_bidirectional = True,
        rnn_stateful = False,
        rnn_layer_norm = True,
        
        # Attention
        attn_type = 'dot_product',
        attn_size = 100,
        attn_dropout = 0,
        attn_normalize = True, 
        attn_activation = 'linear',
        attn_reduction = 'sum', 

        include_status_cnn = False,
        num_filters = 10,
        ngram_filter_sizes = (2, 3),
        
        # CRF
        crf_constraints = None,
        crf_incl_start_end = True,
        crf_reduction = 'sum',
        
        # Logging
        log_dir = None,
        log_subfolder = True,
        
        # Training
        max_len = 50,
        num_epochs_joint = 100,
        num_epochs_sep = 10,
        batch_size = 50,
        num_workers = 6,
        learning_rate = 0.005,
        grad_max_norm = 1,
        overall_reduction = 'sum',

        # Input processing
        pad_start = True,
        pad_end = True,
        
        
        ):
        super(MultitaskEstimator, self).__init__()


        


        # Labels
        self.source = source
        self.event_types = event_types
        self.label_def = label_def
        
        # Word embeddings     
        self.word_embed_source = word_embed_source
        self.word_embed_dir = word_embed_dir

        self.xfmr_name = xfmr_name
        self.xfmr_type = xfmr_type
        self.xfmr_dir = xfmr_dir
        self.use_xfmr = use_xfmr

        
        # Recurrent layer
        self.rnn_type = rnn_type
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bias = rnn_bias
        self.rnn_input_dropout = rnn_input_dropout
        self.rnn_layer_dropout = rnn_layer_dropout
        self.rnn_output_dropout = rnn_output_dropout
        
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_stateful = rnn_stateful
        self.rnn_layer_norm = rnn_layer_norm
        self.rnn_batch_first = True 
        

        
        # Attention
        self.attn_type = attn_type
        self.attn_size = attn_size
        self.attn_dropout = attn_dropout
        self.attn_normalize = attn_normalize 
        self.attn_activation = attn_activation
        self.attn_reduction = attn_reduction
        self.attn_pred_as_seq = True 
        self.include_status_cnn = include_status_cnn
        self.num_filters = num_filters
        self.ngram_filter_sizes = ngram_filter_sizes
        
        # CRF
        self.crf_constraints = crf_constraints
        self.crf_incl_start_end = crf_incl_start_end
        self.crf_reduction = crf_reduction
        
        # Logging
        self.log_dir = log_dir
        self.log_subfolder = log_subfolder
        
        # Training
        self.max_len = max_len
        self.num_epochs_joint = num_epochs_joint
        self.num_epochs_sep = num_epochs_sep
        self.batch_size = batch_size
        self.num_workers = 0 #num_workers
        self.learning_rate = learning_rate
        self.grad_max_norm = grad_max_norm
        self.overall_reduction = overall_reduction               
        
        # Input processing
        self.pad_start = pad_start
        self.pad_end = pad_end

        
        # Number of tags per label
        _, _, self.num_tags = get_label_map(self.label_def)



        self.word_embed_dict = {}
        if self.use_xfmr:
            self.word_embed_dict['map'] = None    
            self.word_embed_dict['matrix'] = None
        else:
                        
            # Get word map and embedding matrix
            self.word_embed_dict['map'], self.word_embed_dict['matrix'] = get_w2v_embed( \
                        model_dir = self.word_embed_dir, 
                        to_pytorch = True, 
                        freeze = True)
         
        # Recurrent layer
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
        
        # Dictionary repository for output layers
        # NOTE: ModuleDict is an ordered dictionary
        self.out_layers = nn.ModuleDict()
        for name, lab_def in self.label_def.items():

            logging.info("")
            logging.info("-"*72)
            logging.info(name)
            logging.info("-"*72)
            
            lab_type = lab_def[LAB_TYPE]
            event_types = list(self.num_tags[name].keys())
            num_tags = self.num_tags[name]            

            # Sentence-level labels            
            if lab_type == SENT:

                
                # Probability vectors from previous
                if name == TRIGGER:
                    seq_feat_size = None
                elif name in [STATUS, TYPE]:
                    seq_feat_size = self.out_layers[TRIGGER].prob_vec_size
                else:
                    seq_feat_size = self.out_layers[STATUS].prob_vec_size + self.out_layers[TYPE].prob_vec_size
                    
                self.out_layers[name] = MultitaskAttention( \
                        event_types = event_types, 
                        num_tags = num_tags, 
                        embed_size = self.rnn.output_size, 
                        vector_size = self.rnn.output_size,
                        seq_feat_size = seq_feat_size,
                        type_ = self.attn_type,
                        dropout = self.attn_dropout,
                        normalize = self.attn_normalize,
                        activation = self.attn_activation,
                        reduction = self.attn_reduction,
                        pred_as_seq = self.attn_pred_as_seq
                        )
            
            
            # cab token-level labels
            elif lab_type == SEQ:
                
                embed_size = self.rnn.output_size + \
                             self.out_layers[STATUS].prob_vec_size + self.out_layers[TYPE].prob_vec_size
                self.out_layers[name] = MultitaskCRF( \
                        event_types = event_types, 
                        num_tags = num_tags, 
                        embed_size = embed_size,
                        constraints = self.crf_constraints,
                        incl_start_end = self.crf_incl_start_end, 
                        reduction = self.crf_reduction)
        
            else:
                raise ValueError("invalid label type:\t{}".format(lab_type))
    
        
        #self.state_dict_exclusions = ['word_embed_matrix.weight']

    def forward(self, X, mask, y=None):
        '''
        
        
        
        Parameters
        ----------
        X: input sequence, already mapped to embeddings
        
        '''
                        
        '''
        Input layer
        '''
        H = self.rnn(X, mask)
       
        
        '''
        Output layers
        '''
        
        # Predictions
        pred = OrderedDict()
        
        # Probability vector across all event types
        prob = OrderedDict()
        
        # Reduced loss across event types (average or sum)
        loss = OrderedDict()
        
        # Iterate over output layers
        for name, layer in self.out_layers.items():

            # Probability vectors from previous
            if name == TRIGGER:
                seq_feats = None
            elif name in [STATUS, TYPE]:
                seq_feats = prob[TRIGGER].detach()
            else:
                seq_feats = torch.cat((prob[STATUS], prob[TYPE]),-1).detach()
            
            pred[name], prob[name], loss[name] = layer( \
                                    X = H, 
                                    y = None if y is None else y[name], 
                                    mask = mask,
                                    seq_feats = seq_feats)

        return (pred, loss)

    

    def fit(self, X, y, device=None):
        '''
        Train multitask model
        '''

        # Get/set device
        device = get_device(device)
        self.to(device)
        
        # Configure training mode
        self.train()
        logging.info('Fitting model')
        logging.info('self.training = {}'.format(self.training))

        # Print model summary
        self.get_summary()

        # Create data set
        dataset = MultitaskDataset( \
                                X = X, 
                                y = y, 
                                label_def = self.label_def, 
                                word_embed_map = self.word_embed_dict['map'],                               
                                word_embed_matrix = self.word_embed_dict['matrix'],
                                xfmr_type = self.xfmr_type,
                                xfmr_dir = self.xfmr_dir,
                                xfmr_device = device,
                                use_xfmr = self.use_xfmr,
                                max_len = self.max_len, 
                                pad_start = self.pad_start, 
                                pad_end = self.pad_end,
                                num_workers = self.num_workers,
                                )
                                        
        # Create data loader
        dataloader = data_utils.DataLoader(dataset, \
                                batch_size = self.batch_size, 
                                shuffle = True, 
                                num_workers = self.num_workers)

        # Create optimizer
        optimizer = optim.Adam(self.parameters(), \
                                                lr = self.learning_rate)

        # Create logger        
        self.writer = create_Tensorboard_writer( \
                                    dir_ = self.log_dir, 
                                    use_subfolder = self.log_subfolder)

        # Loop on epochs
        num_epochs_tot = self.num_epochs_joint + self.num_epochs_sep*len(self.label_def)
        pbar = tqdm(total=num_epochs_tot)
        j_bat = 0
        for j in range(num_epochs_tot):
            
            loss_epoch_tot = 0
            loss_epoch_sep = OrderedDict([(k, 0) for k in self.label_def])
            grad_norm_orig = 0
            grad_norm_clip = 0
            
            # Loop on mini-batches
            for i, (X_, mask_, y_) in  enumerate(dataloader):

                # Reset gradients
                self.zero_grad()
                
                X_ = X_.to(device)
                mask_ = mask_.to(device)
                
                for K, V in y_.items():
                    for k, v in V.items():
                        V[k] = v.to(device)
                

                # Push data through model
                pred, loss = self(X=X_, mask=mask_, y=y_)           
                                

                # Total loss across trigger, status, and entity
                if j < self.num_epochs_joint:
                    loss_bat_tot = loss_reduction(loss, self.overall_reduction)
                else:
                    quotient, _ = divmod(j - self.num_epochs_joint, self.num_epochs_sep)
                    loss_bat_tot = loss[list(loss.keys())[quotient]]
                                        
                # Back probably
                loss_bat_tot.backward()
                grad_norm_orig += clip_grad_norm_(self.parameters(), \
                                                     self.grad_max_norm)
                grad_norm_clip += clip_grad_norm_(self.parameters(), \
                                                           100000000.0)
                optimizer.step()
    
                if self.writer is not None:
                    self.writer.add_scalar('loss_batch', loss_bat_tot, j_bat)        
                j_bat += 1

                # Epoch loss
                loss_epoch_tot += loss_bat_tot.item()
                for k in loss_epoch_sep:
                    loss_epoch_sep[k] += loss[k]

                

            # Average across epochs
            loss_epoch_tot = loss_epoch_tot/i
            for k in loss_epoch_sep:
                loss_epoch_sep[k] += loss_epoch_sep[k]/i
            grad_norm_orig = grad_norm_orig/i
            grad_norm_clip = grad_norm_clip/i

            msg = []
            msg.append('epoch={}'.format(j))
            msg.append('{}={:.1e}'.format('Total', loss_epoch_tot))
            for k, v in loss_epoch_sep.items():
                msg.append('{}={:.1e}'.format(k, v))
            msg = ", ".join(msg)            
            pbar.set_description(desc=msg)
            pbar.update()

            # https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/imagenet/main.py
            if self.writer is not None:
                self.writer.add_scalar('loss_epoch_tot', loss_epoch_tot, j)        
                for k, v in loss_epoch_sep.items():
                    self.writer.add_scalar('loss_{}'.format(k), v, j)        
                self.writer.add_scalar('grad_norm_orig', grad_norm_orig, j)         
                self.writer.add_scalar('grad_norm_clip', grad_norm_clip, j)         
                    
        pbar.close()


    def predict(self, X, device=None):
        '''
        Train multitask model
        '''

        batch_size_pred = 200

        # Get/set device
        device = get_device(device)
        self.to(device)

        # Configure training mode
        self.eval()
        logging.info('Evaluating model')
        logging.info('self.training = {}'.format(self.training))
        

        # Print model summary
        self.get_summary()

        # Create data set
        dataset = MultitaskDataset( \
                                X = X, 
                                y = None, 
                                label_def = self.label_def, 
                                word_embed_map = self.word_embed_dict['map'],                               
                                word_embed_matrix = self.word_embed_dict['matrix'],
                                xfmr_type = self.xfmr_type,
                                xfmr_dir = self.xfmr_dir,
                                xfmr_device = device,
                                use_xfmr = self.use_xfmr,
                                max_len = self.max_len, 
                                pad_start = self.pad_start, 
                                pad_end = self.pad_end,
                                num_workers = self.num_workers,
                                )
        
        # Create data loader
        dataloader = data_utils.DataLoader(dataset, \
                                batch_size = batch_size_pred, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        # Loop on mini-batches
        use_pbar = logging.INFO >= logging.root.level
        
        if use_pbar:
            total = int(sum([len(doc) for doc in X])/self.batch_size)+1
            pbar = tqdm(total=total)
        y_pred = []
        
        for i, (X_, mask_) in enumerate(dataloader):

            X_ = X_.to(device)
            mask_ = mask_.to(device)

            # Push data through model
            pred, loss = self(X=X_, mask=mask_)           

            # Convert from pytorch tensor to list
            for K, V in pred.items():
                for k, v in V.items():
                    if isinstance(v, torch.Tensor):
                        V[k] = v.tolist()
            
            # Convert to list of nested dict
            pred = nested_dict_to_list(pred)
            
            # Append over batches
            y_pred.extend(pred)

            if use_pbar:
                pbar.update()
        
        if use_pbar:
            pbar.close()


        # Post process predicitons
        events = dataset.decode_(y_pred)

        return events

    def get_summary(self):
                
        # Print model summary
        logging.info("\n")
        logging.info("Model summary")
        logging.info(self)
        
        # Print trainable parameters
        logging.info("\n")
        logging.info("Trainable parameters")
        summary = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                summary.append((name, param.size(), mem_size(param)))       
        df = pd.DataFrame(summary, columns=['Param', 'Dim', 'Mem size'])
        logging.info('\n{}'.format(df))
        
        logging.info("\n")
        num_p = sum(p.numel() for p in self.parameters() \
                                                     if p.requires_grad)
        num_pM = num_p/1e6
        logging.info("Total trainable parameters:\t{:.1f} M".format(num_pM))
        logging.info("\n")
        