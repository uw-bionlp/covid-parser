import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid, ReLU
from models.utils import one_hot, one_hot, get_activation_fn, trig_exp, entity_exp, device_check, mem_size, tensor_summary

from allennlp.modules import FeedForward
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.span_extractors import EndpointSpanExtractor
from collections import OrderedDict
from pytorch_memlab import profile
import math

from constants import STATUS, ENTITIES

GOLD = 'gold'
RULE = 'rule'
LEARNED = 'learned'


def prune_arguments(arg_labels, trig_indices_top, arg_indices_top,
                                   trig_mask_top, arg_mask_top):
    """
    
    
    Copied from: https://github.com/dwadden/dygie/blob/master/models/events.py
    
    
    Parameters
    ----------
    arg_labels: argument labels (batch_size, num_trig, num_arg)
    trig_indices_top: indices of trigger spans (batch_size, top_trig_k)
    arg_indices_top: indices of argument spans (batch_size, top_arg_k)
    trig_mask_top: mask for top trigger span indices (batch_size, top_trig_k)
    arg_mask_top: mask for half argument span indices '(batch_size, top_arg_k)

    """
    
    # Get dimensions
    batch_size, top_trig_k = tuple(trig_indices_top.shape)
    batch_size, top_arg_k = tuple(arg_indices_top.shape)

    # Loop on sequences in batch
    arg_labels_pruned = []
    mask_pruned = []

    for labs, trig_ixs, arg_ixs, trig_mask, arg_mask in  \
                zip(arg_labels, trig_indices_top, arg_indices_top,
                                           trig_mask_top, arg_mask_top):
        
        # Dimensionality        
        # labs (num_spans, num_spans)
        # trig_ixs (top_trig_k)
        # arg_ixs (top_arg_k)
        # trig_mask (top_trig_k)
        # arg_mask (top_arg_k)
        
        # Reshape for masking
        trig_mask_exp = trig_mask.unsqueeze(1).repeat(1, top_arg_k)
        arg_mask_exp = arg_mask.unsqueeze(0).repeat(top_trig_k, 1)
        mask = trig_mask_exp*arg_mask_exp
        
        # Prune labels and apply mask
        # (top_trig_k, top_arg_k)     
        labs_pruned = labs[trig_ixs][:, arg_ixs]*mask
        
        # Build batch-level results
        arg_labels_pruned.append(labs_pruned)
        mask_pruned.append(mask)

    arg_labels_pruned = torch.stack(arg_labels_pruned, dim=0)
    mask_pruned = torch.stack(mask_pruned, dim=0)
    
    return (arg_labels_pruned, mask_pruned)




def close_ent(dist, num_entity, mask): 
    '''
    Find closest entity to each trigger
    
    Parameters
    ----------
    dist: distance between tensors with shape (batch_size, num_trig, num_entity)        
    '''
    
    # Indices of minimum distance
    # (batch_size, num_trig) of type LongTensor
    _, indices = dist.min(2)
    
    # Convert to one hot encoding
    # (batch_size, num_trig, num_entity)
    # (batch_size, num_trig) of type LongTensor
    indicator = one_hot(indices, num_entity)

    # Apply mask
    indicator = indicator*mask

    return indicator

def close_trig(dist, num_trig, mask):
    '''
    Find closest trigger to each entity
    '''
    
    # Indices of minimum distance
    _, indices = dist.min(1)
    
    # Convert to one hot encoding
    indicator = one_hot(indices, num_trig)
    
    # Permute to adjust dimensions
    indicator = indicator.permute(0, 2, 1)

    # Apply mask
    indicator = indicator*mask

    return indicator


    
def mask_for_min(X, mask): 
    '''
    Function for masking when finding minimum
    '''
    device = X.device
    return X*mask.type(torch.FloatTensor).to(device) + \
                   1000*(mask != 1.0).type(torch.FloatTensor).to(device)

def distance_features(trig_scores, trig_spans, trig_mask,
                      entity_scores, entity_spans, entity_mask,
                      feature_type=None):
    '''
    Extract distance features for argument spans
    
    
    '''                        

    # Get device
    device = trig_scores.device    
    
    # Get dimensionality            
    batch_size, num_trig, trig_tags = tuple(trig_scores.shape)
    batch_size, num_entity, entity_tags = tuple(entity_scores.shape)

    # Get span start, midpoint, and end
    # (batch_size, num_trig)
    trig_start = trig_spans.type(torch.FloatTensor)[:,:,0].to(device)
    trig_end = trig_spans.type(torch.FloatTensor)[:,:,1].to(device)
    trig_mid = trig_spans.type(torch.FloatTensor).mean(dim=2).to(device)

    # (batch_size, num_entity)
    entity_start = entity_spans.type(torch.FloatTensor)[:,:,0].to(device)
    entity_end = entity_spans.type(torch.FloatTensor)[:,:,1].to(device)
    entity_mid = entity_spans.type(torch.FloatTensor).mean(dim=2).to(device)

    # Get predictions
    # (batch_size, num_trig)
    _, trig_pred = trig_scores.max(-1) 
    # (batch_size, num_entity)
    _, entity_pred = entity_scores.max(-1)
    
    # Repeat trigger and entity tensor so same dimension
    # (batch_size, num_trig, num_entity)
    trig_pred  = trig_exp(trig_pred,  num_entity)
    trig_start = trig_exp(trig_start, num_entity)
    trig_end   = trig_exp(trig_end,   num_entity)
    trig_mid   = trig_exp(trig_mid,   num_entity)
    trig_mask  = trig_exp(trig_mask,  num_entity)
    
    entity_pred  = entity_exp(entity_pred,  num_trig)
    entity_start = entity_exp(entity_start, num_trig)
    entity_end   = entity_exp(entity_end,   num_trig)
    entity_mid   = entity_exp(entity_mid,   num_trig)
    entity_mask  = entity_exp(entity_mask,  num_trig)

    # Create masks
    # (batch_size, num_trig, num_entity)
    # Only include valid triggers
    #trig_mask = trig_mask.type(torch.FloatTensor)
    # Only include valid entities
    #entity_mask = entity_mask.type(torch.FloatTensor)
    # Only include non-null trigger labels
    trig_pred_mask = (trig_pred > 0).type(torch.LongTensor).to(device)

    # Only include non-null entity labels
    entity_pred_mask = (entity_pred > 0).type(torch.LongTensor).to(device)
    
    # Merge masks
    # (batch_size, num_trig, num_entity)
    mask = trig_mask*entity_mask*trig_pred_mask*entity_pred_mask
   
    # Binary indicator of overlap
    # Note that end indices are exclusive, which accounts for the 
    # decrementing of the end indices
    overlap = ((trig_start <= entity_end - 1)*(entity_start <= trig_end - 1)).type(torch.LongTensor).to(device)*mask

    # Distance between trigger and entity spans
    # (batch_size, num_trig, num_entity)
    # Distance between spans
    dist = entity_mid - trig_mid
    # Set masked distances to 0 and replace with large number
    dist = mask_for_min(dist, mask)
    # Is entity after or before trigger?
    is_before = (dist < 0).type(torch.LongTensor).to(device)
    is_after = (dist > 0).type(torch.LongTensor).to(device)

    # Masked distances indicating before or after trigger
    dist_before = mask_for_min(dist, is_before)
    dist_after = mask_for_min(dist, is_after)
    
    # Distance feature dictionary
    close_ = OrderedDict() 

    # Find closest entity to trigger, as binary indicator
    # (batch_size, num_trig, num_entity)        
    close_['entity_anywhere']    = close_ent(dist.abs_(), num_entity, mask)
    close_['entity_before'] = close_ent(dist_before, num_entity, mask)
    close_['entity_after']  = close_ent(dist_after, num_entity, mask)
    
    # Find closest trigger to entity, as binary indicator
    # (batch_size, num_trig, num_entity)        
    close_['trig_anywhere']    = close_trig(dist.abs_(), num_trig, mask)
    close_['trig_before'] = close_trig(dist_before, num_trig, mask)
    close_['trig_after']  = close_trig(dist_after, num_trig, mask)

    # Build and return feature vector
    if feature_type is None:
        # Create feature vector
        # (batch_size, num_trig, num_entity, 6)
        features = [v for k, v in close_.items()]
        features = torch.stack(features, dim=3).type(torch.FloatTensor).to(device)
        return features
    else:
        return close_[feature_type]
    
    
class ArgumentScorerGold(nn.Module):
    '''
    Argument scorer using gold labels
    Convert labels to one hot encoding
    
    
    Parameters
    ----------
    num_tags: label vocab size
    
    
    Returns
    -------
    arg_scores: tensor of scores (batch_size, num_trig, arg_num, num_tags)
    
    '''
    def __init__(self, num_tags=2, low_val=-5, high_val=5):
        super(ArgumentScorerGold, self).__init__()
            
        self.num_tags = num_tags
        self.low_val = low_val
        self.high_val = high_val

    def forward(self, arg_labels):
        '''
        Parameters
        ----------
        arg_labels: tensor of labels (batch_size, num_trig, arg_num)
        
        Returns
        -------
        arg_scores: tensor of scores (batch_size, num_trig, arg_num, num_tags)
        '''
        
        # Argument scores
        arg_scores = one_hot(arg_labels, self.num_tags, \
                                             low_val = self.low_val, 
                                             high_val = self.high_val)
        
        return arg_scores.type(torch.FloatTensor)


class ArgumentScorerRule(nn.Module):
    '''
    Argument scorer using rule-based approach, based on span relative 
    positions
    
    
    Parameters
    ----------
    mode: indicates whether STATUS or ENTITIES
        If mode == STATUS, then find closest non-null status span, for each non-null trigger
        If mode == ENTITIES, then find the closest, up-stream non-null trigger span, for each non-null entity
    
    
    Returns
    -------
    arg_scores: tensor of scores (batch_size, num_trig, arg_num, 2)
    
    '''
    def __init__(self, mode, low_val=-5, high_val=5):
        super(ArgumentScorerRule, self).__init__()
            
        self.mode = mode
        self.low_val = low_val
        self.high_val = high_val
        
    def forward(self, trig_scores, trig_spans, trig_mask,
                      entity_scores, entity_spans, entity_mask):
        '''
        Parameters
        ----------
        trig_scores: tensor of scores (batch_size, num_trig, trig_tags)
        trig_spans: tensor of spans (batch_size, num_trig, 2)
        trig_mask: tensor of mask (batch_size, num_trig)
        entity_scores: tensor of scores (batch_size, num_entity, entity_tags)
        entity_spans: tensor of spans (batch_size, num_entity, 2)
        entity_mask: tensor of mask (batch_size, num_entity)        
        
        Returns
        -------         
        arg_scores: tensor of scores (batch_size, num_trig, num_entity, 2)        
        '''
            
        # Number of argument tags (vocab size) is 2 because binary
        arg_tags = 2 

        # Check mode
        assert self.mode in [STATUS, ENTITIES]

        # Minimum distance and one hot encoding
        if self.mode == STATUS:
            feature_type = 'entity_anywhere'

        elif self.mode == ENTITIES:
            feature_type = 'trig_before'
        
        # Get closest span
        closest = distance_features(
                            trig_scores = trig_scores, 
                            trig_spans = trig_spans, 
                            trig_mask = trig_mask,
                            entity_scores = entity_scores, 
                            entity_spans = entity_spans, 
                            entity_mask = entity_mask,
                            feature_type = feature_type)

        # Convert to one hot encoding for scores
        # i.e. making hard decisions
        arg_scores = one_hot(closest, arg_tags, \
                                             low_val = self.low_val, 
                                             high_val = self.high_val)

        return arg_scores.type(torch.FloatTensor)

class ArgumentScorerLearned(nn.Module):
    '''
    Argument scorer learned using feedforward neural networks
    
    
    Parameters
    ----------
    num_tags: label vocab size
    
    Returns
    -------
    arg_scores: tensor of scores (batch_size, num_trig, arg_num, 2)
    
    '''
    def __init__(self, input_dim, hidden_dim, \
            output_dim = 2,
            activation = 'relu',
            dropout = 0.0
            ):
            
            
        super(ArgumentScorerLearned, self).__init__()
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout

        self.num_layers = 1

        # Feedforward neural network
        self.ffnn = FeedForward( \
                    input_dim = self.input_dim,
                    num_layers = self.num_layers,
                    hidden_dims = self.hidden_dim, 
                    activations = self.activation_fn,
                    dropout = self.dropout)

        # Score (linear projection)                    
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)

    #@profile
    def forward(self, trig_embed, arg_embed, additional_feat=None):
        '''
        Parameters
        ----------
        trig_label_scores: tensor of scores (batch_size, num_trig, trig_tags)
        trig_spans: tensor of spans (batch_size, num_trig, 2)
        trig_embed: tensor of embeddings (batch_size, num_trig, embed_dim)
        entity_scores: tensor of scores (batch_size, num_entity, entity_tags)
        entity_spans: tensor of spans (batch_size, num_entity, 2)
        entity_mask: tensor of mask (batch_size, num_entity)        
        arg_embed: tensor of embeddings (batch_size, num_entity, embed_dim)
                
        Returns
        -------         
        arg_scores: tensor of scores (batch_size, num_trig, num_entity, output_dim)        
        '''


        
        device_check([trig_embed, arg_embed])
                
        device = trig_embed.device                

        # Get size
        batch_size, num_trig, embed_dim = tuple(trig_embed.shape)       
        batch_size, num_entity, embed_dim = tuple(arg_embed.shape)

        '''
        Create trigger-entity embedding pairs
        '''    
        # Expand and tile trigger
        # (batch_size, num_trig, num_entity, embed_dim)
        trig_embed = trig_embed.unsqueeze(2).repeat(1, 1, num_entity, 1)       

        # Expand and tile entity
        # (batch_size, num_trig, num_entity, embed_dim)
        arg_embed = arg_embed.unsqueeze(1).repeat(1, num_trig, 1, 1)

        # Concatenate trigger and entity embeddings
        # (batch_size, num_trig, num_entity, embed_dim*2)
        embeddings = torch.cat((trig_embed, arg_embed), dim=3)
        
        # Include distance features
        if additional_feat is not None:
            embeddings = torch.cat((embeddings, additional_feat), dim=3)

        '''
        Score trigger-entity embedding pairs
        '''
        pair_embed_dim = embeddings.size(-1)
        
        # Flatten
        embeddings = embeddings.view(-1, pair_embed_dim)
        
        # Push masked embeddings through feedforward neural network
        projected = self.ffnn(embeddings)
        arg_scores = self.linear(projected)

        # Inflate
        # (batch_size, num_trig, num_entity, num_tags)
        arg_scores = arg_scores.view(batch_size, num_trig, \
                                            num_entity, self.output_dim)
        
        return arg_scores





   