
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from multiprocessing import Pool 
import numpy as np
import time

from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules import FeedForward
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules import TimeDistributed, Pruner
from allennlp.nn import util

from models.utils import one_hot, get_activation_fn, map_dict_builder, one_hot, get_predictions, get_device, create_mask, map_dict_builder
from models.training import get_loss
from models.crf import BIO_to_span
from pytorch_memlab import profile

from constants import *


NEG_FILL = -1e20



def seq_tags_to_spans(seq_tags, span_map, tag_to_lab_fn):
    '''
    Convert sequence tags to span labels
    
    Parameters
    ----------
    seq_tags: list of list of label indices
             i.e. list of sentences, where each sentence 
                  is a list of label indices
    
    Returns
    -------
    span_labels: tensor of shape (batch_size, num_spans)
    
    '''

    #start_time = time.time()

    # Get inputs for tensor initialization
    batch_size = len(seq_tags)
    num_spans = len(span_map)

    # Initialize span labels to null label
    span_labels = torch.zeros(batch_size, num_spans).type( \
                                                   torch.LongTensor)

    # Loop on sequences
    for i_seq, seq in enumerate(seq_tags):
         
        # Convert BIO to spans
        S = BIO_to_span(seq, \
                        lab_is_tuple=False, 
                        tag_to_lab_fn = tag_to_lab_fn)
        
        # Iterate over spans
        for lab, start, end in S:
            
            # Token indices of current span
            idx = (start, end)
            
            # Span in map
            if idx in span_map:

                # Span index within tensor
                i_span = span_map[idx]
                
                # Update label tensor
                span_labels[i_seq, i_span] = lab
            
            # Span not in map
            else:
                logging.warn("span not in map:\t{}".format(idx))

    #print("--- %s ms ---" % ((time.time() - start_time)*1000))
    return span_labels


def num_keep_from_one_hot(X):
    '''
    Get a number to keep from one hot encoding
    '''
    
    assert X.dim() == 3
    
    # Sum one hot values across last dimension
    # Find nonnegative values
    # Get counts for each batch
    X = X.sum(-1).gt(0).sum(-1)
    
    X = torch.max(X, torch.ones_like(X))
    
    return X
    
    

class SpanScorerGold(nn.Module):
    '''
    Span scorer using gold labels
    Convert labels to one hot encoding
    
    
    Parameters
    ----------
    num_tags: label vocab size
    

    
    '''
    def __init__(self, num_tags, low_val=-5, high_val=5):
        super(SpanScorerGold, self).__init__()
            
        self.num_tags = num_tags
        self.low_val = low_val
        self.high_val = high_val

    def forward(self, labels):
        '''
        Parameters
        ----------
        
        
        Returns
        -------
        span_scores: tensor of size (batch_size, num_span, num_tags)
        '''
       
        logits = one_hot(labels, \
                    num_tags = self.num_tags, 
                    low_val = self.low_val, 
                    high_val = self.high_val).float()
        return logits



class SpanScorerCRF(nn.Module):
    '''
    Span extractor
    '''
    def __init__(self, embed_size, tag_to_span_fn, num_tags_seq, num_tags_span,
            low_val = -5, 
            high_val = 5, 
            constraints = None,
            incl_start_end = True,
            name = None,
            timers = None,
            limit_decode = False,
            ):
        super(SpanScorerCRF, self).__init__()

        self.embed_size = embed_size
        self.tag_to_span_fn = tag_to_span_fn
        self.num_tags_seq = num_tags_seq
        self.num_tags_span = num_tags_span
        self.low_val = low_val
        self.high_val = high_val       
        self.constraints = constraints
        self.incl_start_end = incl_start_end
        self.name = name
        self.timers = timers
        self.limit_decode = limit_decode
                
        # Linear projection layer
        self.projection = nn.Linear(embed_size, self.num_tags_seq)

        # Create event-specific CRF
        self.crf = ConditionalRandomField( \
                        num_tags = self.num_tags_seq, 
                        constraints = constraints,
                        include_start_end_transitions = incl_start_end)            

    def forward(self, seq_tensor, seq_mask, span_map, span_indices, \
                                                    seq_labels=None):
        '''
        Calculate logits 
        '''
        # Dimensionality
        batch_size, max_seq_len, input_dim = tuple(seq_tensor.shape)

        # Project input tensor sequence to logits
        self.timers.start('{}:\tCRF - projection'.format(self.name))
        logits = self.projection(seq_tensor)
        self.timers.stop('{}:\tCRF - projection'.format(self.name))

        '''
        Calculate loss
        '''
        # No labels provided
        if seq_labels is None:
            loss = None
            
        # Span labels provided (i.e. training)
        else:           
                                            
            # Get loss (negative log likelendB)
            self.timers.start('{}:\tCRF - loss'.format(self.name))
            loss = -self.crf( \
                                inputs = logits, 
                                tags = seq_labels, 
                                mask = seq_mask)
            self.timers.stop('{}:\tCRF - loss'.format(self.name))

        '''
        Decoding sequence tags
        '''        
        
        
        # Best path
        self.timers.start('{}:\tCRF - decoding'.format(self.name))

        # Only decode sequences with at least one positive label, 
        # per logits (does not account for transition scores).
        # Admittedly, this is not entirely mathematically correct and 
        # may lead to a reduction in recall; however, the "limit_decode" 
        # mode is fine for hyperparameter tuning.
        if self.limit_decode:
            
            # Indices of logits maximum
            # (batch_size, max_seq_len)
            logits_max_idx = logits.argmax(-1)
            
            # Has non-null prediction
            # (batch_size)
            non_null = logits_max_idx.sum(-1) > 0
            
            # Indices with positive label predictions
            non_null_idx = non_null.int().nonzero().squeeze().tolist()
            if isinstance(non_null_idx, int):
                non_null_idx = [non_null_idx]


            # At least one sequence needs decoding
            if len(non_null_idx) > 0:

                # Viterbi decode
                best_paths = self.crf.viterbi_tags( \
                                            logits = logits[non_null], 
                                            mask = seq_mask[non_null])
                seq_pred, score = zip(*best_paths)
                seq_pred = list(seq_pred)

            # No sequences need decoding
            else:
                seq_pred = [[] for _ in range(batch_size)]
       
            # Need to reindex, so all sequences present, 
            # if not decoded            
            
            # Initialize output predictions
            seq_pred_tmp = [[] for _ in range(batch_size)]            
            
            # Iterate over sequences with likely positive label
            for idx, pred in zip(non_null_idx, seq_pred):
                seq_pred_tmp[idx] = pred
            seq_pred = seq_pred_tmp

        # Decode all sequences, regardless of logits
        else:
            
            # Viterbi decode
            best_paths = self.crf.viterbi_tags( \
                                            logits = logits, 
                                            mask = seq_mask)
            seq_pred, score = zip(*best_paths)
            seq_pred = list(seq_pred)
        
        self.timers.stop('{}:\tCRF - decoding'.format(self.name))

        '''
        Convert sequence tags to span predictions
        '''
        self.timers.start('{}:\tCRF - seq to spans'.format(self.name))
        # Get spans from sequence tags
        #   Converts list of list of predicted label indices to
        #   tensor of size (batch_size, num_spans)
        span_pred = seq_tags_to_spans( \
                                seq_tags = seq_pred, 
                                span_map = span_map, 
                                tag_to_lab_fn = self.tag_to_span_fn)
        
        span_pred = span_pred.to(seq_tensor.device)

        # Get scores from labels
        span_pred = F.one_hot(span_pred, num_classes=self.num_tags_span).float()
        self.timers.stop('{}:\tCRF - seq to spans'.format(self.name))
        


        return (span_pred, loss)








class SpanScorerFFNN(nn.Module):
    '''
    Span scorer 
    
    
    Parameters
    ----------
    num_tags: label vocab size
    
    
    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num, num_tags)
    
    '''
    def __init__(self, input_dim, hidden_dim, num_tags, \
            activation = 'relu',
            dropout = 0.0,
            loss_reduction = 'sum',
            seq_feat_size = None):
        super(SpanScorerFFNN, self).__init__()
            

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
       
        self.activation = activation
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.loss_reduction = loss_reduction
        self.seq_feat_size = seq_feat_size
        
        self.num_layers = 1
        
        '''
        Create classifier
        '''
        
        if self.seq_feat_size is not None:
            self.input_dim += self.seq_feat_size
            
        
        # Feedforward neural network for predicting span labels
        self.FF = FeedForward( \
                    input_dim = self.input_dim,
                    num_layers = self.num_layers,
                    hidden_dims = self.hidden_dim, 
                    activations = self.activation_fn,
                    dropout = self.dropout)

        # Span classifier
        self.scorer = torch.nn.Sequential(
            TimeDistributed(self.FF),
            TimeDistributed(torch.nn.Linear(self.hidden_dim, self.num_tags)))

    def forward(self, embed, mask, labels=None, seq_feat=None, verbose=False):

        '''
        Parameters
        ----------
        embed: (batch_size, num_spans, input_dim)        
        mask: span mask (batch_size, num_spans)        
        labels: true span labels (batch_size, num_spans)  
        
        seq_feat: sequence-level features (batch_size, seq_feat_size)      
        
        Returns
        -------
        '''

        if verbose:
            logging.info('Span scorer, forward')       
            logging.info('\tembed, original:\t{}'.format(embed.shape))                
                       
        if seq_feat is not None:
            batch_size, num_spans, input_dim = tuple(embed.shape)
        
        
            seq_feat_rep = seq_feat.unsqueeze(1).repeat(1, num_spans, 1)

            embed = torch.cat((embed, seq_feat_rep), dim=-1)
        
            if verbose:
                logging.info('\tseq_feat, original:\t{}'.format(seq_feat.shape))                
                logging.info('\tseq_feat, repeated:\t{}'.format(seq_feat_rep.shape))                
                logging.info('\tembed, concat:\t{}'.format(embed.shape))                
       
       
        # Compute label scores
        # (batch_size, num_spans, num_tags)
        logits = self.scorer(embed)

        # Calculate loss
        if labels is None:
            loss = None
        else:
            loss = get_loss( \
                    scores = logits, 
                    labels = labels, 
                    mask = mask,
                    reduction = self.loss_reduction)

        if verbose:
            logging.info('\tlogits:\t{}'.format(logits.shape))                
            
        return (logits, loss)
        








    

    
def get_span_overlaps(spans, end_is_exclusive=True):
    '''
    Determine if spans are overlapping
    
    Parameters
    ----------
    spans: tensor of span indices (batch_size, num_items, 2)
    
    
    Returns
    -------
    overlaps: tensor of Boolean indicating overlap (batch_size, num_items, num_items)
    '''
    
    # Get and check dimensionality
    batch_size, num_items, num_idx = tuple(spans.shape)
    assert num_idx == 2
    
    # Start and end indices repeated across third dimension
    startA = spans[:,:,0].unsqueeze(-1).repeat(1, 1, num_items)
    endA =   spans[:,:,1].unsqueeze(-1).repeat(1, 1, num_items)

    # Decrement end indices, if exclusive (i.e. like Python or C++)        
    if end_is_exclusive:
        endA += -1        
        
    # Start and end indices repeated across second dimension
    startB = startA.transpose(1, 2)
    endB = endA.transpose(1, 2)

    # Assess overlap
    overlaps = (startA <= endB)*(startB <= endA)
    
    return overlaps

def filter_overlapping(logits, mask, spans, fill_val=NEG_FILL):
    '''
    Filter scores based on overlap
    
   
    
    Parameters
    ----------
    logits: tensor label scores (batch_size, num_items, num_tags)
    mask: boolean mask (batch_size, num_items)
    overlap: tensor of Boolean (num_items, num_items)
    fill_val: value to substitute into scores at overlapping locations
            (Likely a large negative number or 0)
    
    
    returns: scores with overlapping span values replaced
    '''

    # Get dimensionality
    batch_size, num_items, num_tags = tuple(logits.shape)

    # Get boolean overlap tensors
    overlaps = get_span_overlaps(spans)
    
    # For indexing across sequences in batch
    idx_batch = np.arange(batch_size)
    
    # Get scores
    # (batch_size, num_items)
    scores = logit_scorer(logits, agg_type='max')

    # Get predictions
    pred = get_predictions(logits, mask)
        
    # Mask scores
    # (batch_size, num_items)
    invalid = ~(mask.bool())
    # (batch_size, num_items)
    scores            
        
    # Mask for assessing maximums
    # 0 = no maximum found at location
    # 1 = maximum found at location
    # (batch_size, num_items)
    max_found = torch.zeros_like(scores).bool()

    # Iterate over items
    for i in range(num_items):

        # Force masked values to large negative number
        # (batch_size, num_items)
        scores_masked = torch.masked_fill(scores, max_found, fill_val)
        
        # Get maximums
        # (batch_size, 1)
        scores_max, scores_idx = scores_masked.max(1)
                
        # Stop iterating if maximum is fill value
        if scores_max.max() == fill_val:
            break
        
        # Set indices of maximums to 1
        max_found[idx_batch, scores_idx] = 1
        
        # Overlap between spans
        # (batch_size, num_items)
        is_overlap = overlaps[idx_batch, scores_idx]
        
        # Prediction for maximum scores
        pred_max = pred[idx_batch, scores_idx]
        
        # Is prediction same as the prediction for the max score?
        is_pred_max = (pred == pred_max.unsqueeze(-1).repeat(1, num_items))
        
        # Only consider valid positions
        is_pred_max = torch.masked_fill(is_pred_max, invalid, 0)

        # Score positions to fill with large negative number
        to_fill = (is_overlap*is_pred_max)*(~max_found)

        # Update scores
        scores = scores.masked_fill_(to_fill, fill_val)
        
        zeros = torch.zeros_like(to_fill).unsqueeze(-1)
        to_fill = to_fill.unsqueeze(-1).repeat(1, 1, num_tags-1)
        to_fill = torch.cat((zeros, to_fill), 2)
        logits = logits.masked_fill_(to_fill, fill_val)
    
    return logits


def logit_scorer(logits, agg_type='max'):
    '''
    Convert logits to single score
    '''
    if agg_type == 'max':
        scores, _ = logits[:,:,1:].max(dim=-1)
    elif agg_type == 'sum':
        scores = logits[:,:,1:].sum(dim=-1)
    else:
        raise ValueError("incorrect agg_type: {}".format(agg_type))
    return scores

#def span_pruner(logits, mask, seq_length, spans_per_word, agg_type='max', \
#                prune_overlapping=False, span_overlaps=None):
def span_pruner(logits, mask, seq_length, spans_per_word=1, num_keep=None, agg_type='max'):
        """
        
        Based on AllenNLP allennlp.modules.Pruner from release 0.84
        
        
        Parameters
        ----------
        
        logits: (batch_size, num_spans, num_tags)
        mask: (batch_size, num_spans)
        num_keep: int OR torch.LongTensor
                If a tensor of shape (batch_size), specifies the 
                number of items to keep for each
                individual sentence in minibatch.
                If an int, keep the same number of items for all sentences.
        
        
        """


        batch_size, num_items, num_tags = tuple(logits.shape)

        
        # Number to keep not provided, so use spans per word
        if num_keep is None:
            num_keep = seq_length*spans_per_word
            num_keep = torch.max(num_keep, torch.ones_like(num_keep))

        # If an int was given for number of items to keep, construct tensor by repeating the value.
        if isinstance(num_keep, int):
            num_keep = num_keep*torch.ones([batch_size], dtype=torch.long,
                                                               device=mask.device)

        # Maximum number to keep
        max_keep = num_keep.max()

        # Get scores from logits
        scores = logit_scorer(logits)
        
        # Set overlapping span scores large neg number
        #if prune_overlapping:
        #    scores = overlap_filter(scores, span_overlaps)
        
        # Add dimension
        scores = scores.unsqueeze(-1)


        # Check scores dimensionality
        if scores.size(-1) != 1 or scores.dim() != 3:
            raise ValueError(f"The scorer passed to Pruner must produce a tensor of shape"
                             f"(batch_size, num_items, 1), but found shape {scores.size()}")
        
        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        mask = mask.unsqueeze(-1)
        scores = util.replace_masked_values(scores, mask, NEG_FILL)

        # Shape: (batch_size, max_num_items_to_keep, 1)
        _, top_indices = scores.topk(max_keep, 1)

        # Mask based on number of items to keep for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices_mask = util.get_mask_from_sequence_lengths(num_keep, max_keep)
        top_indices_mask = top_indices_mask.bool()

        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Fill all masked indices with largest "top" index for that sentence, so that all masked
        # indices will be sorted to the end.
        # Shape: (batch_size, 1)
        fill_value, _ = top_indices.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)
        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size * max_num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        flat_indices = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
        # the top k for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        sequence_mask = util.batched_index_select(mask, top_indices, flat_indices)
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        # Shape: (batch_size, max_num_items_to_keep, 1)
        top_scores = util.batched_index_select(scores, top_indices, flat_indices)
        
        
        # Shape: (batch_size, max_num_items_to_keep)
        top_scores = top_scores.squeeze(-1)
        

        return (top_indices, top_scores, top_mask)


#
#def filter_overlappingXX(logits, mask, spans, fill_val=NEG_FILL):
#    '''
#    Filter scores based on overlap
#    
#   
#    
#    Parameters
#    ----------
#    logits: tensor label scores (batch_size, num_items, num_tags)
#    mask: boolean mask (batch_size, num_items)
#    overlap: tensor of Boolean (num_items, num_items)
#    fill_val: value to substitute into scores at overlapping locations
#            (Likely a large negative number or 0)
#    
#    
#    returns: scores with overlapping span values replaced
#    '''
#
#    # Get dimensionality
#    batch_size, num_items, num_tags = tuple(logits.shape)
#
#    # Get boolean overlap tensors
#    overlaps = get_span_overlaps(spans)
#
#    #    print('overlaps', overlaps.shape)
#    #    
#    #    for i in range(batch_size):
#    #        print(i)
#    #        print(torch.transpose(spans[i], 0, 1))
#    #        for j in range(num_items):
#    #            print(j)
#    #            print([int(x) for x in overlaps[i][j].cpu().numpy().tolist()])
#    #    
#    #    
#    #    x = sldjf
#    
#    # For indexing across sequences in batch
#    idx_batch = np.arange(batch_size)
#    
#    # Get scores
#    scores = logit_scorer(logits, agg_type='max')
#        
#    # Mask scores
#    invalid = ~(mask.bool())
#    scores = torch.masked_fill(scores, invalid, fill_val)            
#        
#    # Mask for assessing maximums
#    # 0 = no maximum found at location
#    # 1 = maximum found at location
#    max_found = torch.zeros_like(scores).bool()
#    print('overlap', overlaps)
#    print('before')
#    print(logits)
#    print('scores')
#    print(scores)
#    
#    print('num_items', num_items)
#    # Iterate over items
#    for i in range(num_items):
#        print('i', i)
#        # Force masked values to large negative number
#        # (batch_size, num_items)
#        scores_masked = torch.masked_fill(scores, max_found, fill_val)
#        
#        # Get maximums
#        # (batch_size, 1)
#        scores_max, scores_idx = scores_masked.max(1)
#                
#        # Stop iterating if maximum is fill value
#        if scores_max.max() == fill_val:
#            print('ALL NEG_FILL '*10)
#            break
#            
#        
#        # Set indices of maximums to 1
#        max_found[idx_batch, scores_idx] = 1
#        
#        # Overlap between spans
#        # (batch_size, num_items)
#        print('overlaps', overlaps.shape)
#        print('scores_idx', scores_idx.shape)
#        is_overlap = overlaps[idx_batch, scores_idx]
#        #is_overlap = torch.index_select(overlaps, 0, scores_idx)
#        
#        
#        print('is_overlap', is_overlap.shape)
#
#        # Score positions to fill with large negative number
#        to_fill = is_overlap*(~max_found)
#        print('to_fill', to_fill.shape, to_fill.dtype)
#        # Update scores
#        scores = scores.masked_fill_(to_fill, fill_val)
#        
#        zeros = torch.zeros_like(to_fill).unsqueeze(-1)
#        to_fill = to_fill.unsqueeze(-1).repeat(1, 1, num_tags-1)
#        print('zeros', zeros.shape)
#        print('to_fill', to_fill.shape)
#        to_fill = torch.cat((zeros, to_fill), 2)
#        logits = logits.masked_fill_(to_fill, fill_val)
#    print('after')
#    print(logits)
#    #z = slkdjf
#    
#    return logits



#def overlap_filter(scores, overlap, fill_val=NEG_FILL):
#    '''
#    Filter scores based on overlap
#    
#   
#    
#    Parameters
#    ----------
#    scores: tensor scores (batch_size, num_items)
#    overlap: tensor of Boolean (num_items, num_items)
#    fill_val: value to substitute into scores at overlapping locations
#            (Likely a large negative number or 0)
#    
#    
#    returns: scores with overlapping span values replaced
#    '''
#
#    # Get dimensionality
#    batch_size, num_items = tuple(scores.shape)
#
#    # Remove batch dim, if present
#    if len(overlap.shape) == 3:
#        overlap = overlap[0]
#
#    # For indexing across sequences in batch
#    idx_batch = np.arange(batch_size)
#    
#    # Mask for assessing maximums
#    # 0 = no maximum found at location
#    # 1 = maximum found at location
#    max_found = torch.zeros_like(scores).bool()
#    
#    # Iterate over items
#    for i in range(num_items):
#        
#        # Force masked values to large negative number
#        # (batch_size, num_items)
#        scores_masked = torch.masked_fill(scores, max_found, fill_val)
#        
#        # Get maximums
#        # (batch_size, 1)
#        scores_max, scores_idx = scores_masked.max(1)
#                
#        # Stop iterating if maximum is fill value
#        if scores_max.max() == fill_val:
#            break
#        
#        # Set indices of maximums to 1
#        max_found[idx_batch, scores_idx] = 1
#        
#        # Overlap between spans
#        # (batch_size, num_items)
#        is_overlap = torch.index_select(overlap, 0, scores_idx)
#        
#        # Score positions to fill with large negative number
#        to_fill = is_overlap*(~max_found)
#        
#        # Update scores
#        scores.masked_fill_(to_fill, fill_val)
#
#    return scores
