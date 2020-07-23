
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import math
from torch.nn.modules.activation import Sigmoid, ReLU
import numpy as np
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules import FeedForward
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules import TimeDistributed, Pruner
from allennlp.nn import util
from allennlp.nn.util import batched_index_select, get_range_vector, get_device_of


import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
from overrides import overrides

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util


from pytorch_models.utils import one_hot, get_activation_fn
from pytorch_models.utils import map_dict_builder
from pytorch_models.utils import create_mask, map_dict_builder
from pytorch_models.training import get_loss
from pytorch_models.crf import BIO_to_span
from pytorch_memlab import profile

#from pytorch_models.span_embedder import positional_feature_plot as p

from constants import *



def get_range_vector_ORIG(size, device):
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)



def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns segmented spans in the target with respect to the provided span indices.
    It does not guarantee element order within each span.
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_ends - max_span_range_indices
    # We also don't want to include span indices which are less than zero,
    # which happens because some spans near the beginning of the sequence
    # have an end index < max_batch_span_width, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices >= 0)
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask







class SpanEmbedder(nn.Module):
    '''
    Create span embeddings
    
    
    Parameters
    ----------
    num_tags: label vocab size
    
    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num, 2)
    
    '''
    def __init__(self, input_dim, \
            
            # Endpoint extractor parameters        
            use_endpoint = True, 
            endpoint_combos = "x,y",
            num_width_embeddings = None,
            span_width_embedding_dim = None,
            
            # Attention parameters
            use_attention = True,
            
            # Heuristic position parameters
            use_position_heuristic = False,
            max_seq_len = None,
            position_hidden_size = None,           
            min_timescale = 1.0, 
            max_timescale = 1.0e4,    
            
            # Learned of position parameters
            use_position_learned = False,
                                                
            # FFNN projection parameters
            project = False,
            hidden_dim = None,
            activation = 'tanh',
            dropout = 0.0,
            
            # General config
            span_end_is_exclusive = True,       
            
            device_as_int = 1,     
            ):
            
        super(SpanEmbedder, self).__init__()

        # Span end is exclusive (like Python or C)
        self.span_end_is_exclusive = bool(span_end_is_exclusive)

           
        # Initialize span dimensionality
        span_dim = 0

        '''
        Endpoint extractor
        '''
        self.use_endpoint = bool(use_endpoint)
        if self.use_endpoint:
          
            # Endpoint combinations            
            endpoint_combos = str(endpoint_combos)
            if span_width_embedding_dim is None:
                    num_width_embeddings = None
            self._endpoint_extractor = EndpointSpanExtractor( \
                    input_dim = input_dim,
                    combination = endpoint_combos,
                    num_width_embeddings = num_width_embeddings,
                    span_width_embedding_dim = span_width_embedding_dim,
                    bucket_widths = False)
            
            
            # Increment span dimensionality
            n = len(endpoint_combos.split(','))
            span_dim += input_dim*n

            # Span width embedding
            if span_width_embedding_dim is not None:
                    span_dim += span_width_embedding_dim            

        '''
        Self-attentive span extractor
        '''
        self.use_attention = bool(use_attention)
        if self.use_attention:
            
            # Create extractor
            self._attentive_extractor = SelfAttentiveSpanExtractor( \
                                input_dim = input_dim)
               
            # Increment span dimensionality
            span_dim += input_dim
                    
        '''       
        Heuristic positional embeddings (based on sine and cosine)
        '''
        self.use_position_heuristic = bool(use_position_heuristic)
        if self.use_position_heuristic:
            embed = positional_features( \
                        timesteps = max_seq_len, 
                        hidden_dim = position_hidden_size,
                        device = device_as_int,
                        min_timescale = min_timescale, 
                        max_timescale = max_timescale)
            self.position_embed = nn.Embedding(embed.size(0), embed.size(1))
            self.position_embed.weight = nn.Parameter(embed)
            self.position_embed.weight.requires_grad = False

            # Increment span dimensionality
            span_dim += position_hidden_size        
        '''
        Learned positional embeddings                
        '''
        self.use_position_learned = bool(use_position_learned)
        if self.use_position_learned:
            raise ValueError("Model currently does not implement learned positional embeddings")
        

        '''
        Nonlinear projection via feedforward neural network
        '''
        self.project = project
        if self.project:                       
            self.ffnn = FeedForward( \
                    input_dim = span_dim,
                    num_layers = 1,
                    hidden_dims = hidden_dim, 
                    activations = get_activation_fn(activation),
                    dropout = dropout)        
            self.output_dim = hidden_dim
        else:
            self.output_dim = span_dim



    def forward(self, sequence_tensor, sequence_mask, 
                      span_indices, span_mask, verbose=False):
        '''
        Parameters
        ----------
        sequence_tensor: sequence representation (batch_size, seq_len, embed_dim)
        sequence_mask: sequence mask (batch_size, seq_len)
        span_indices: tensor of span indices (batch_size, span_num, 2)
        span_mask: tensor of mask (batch_size, trig_num)
                
        Returns
        -------         
        span_embed: tensor of span embeddings (batch_size, span_num, output_dim)        
        '''
            
        if verbose:
            logging.info('')
            logging.info('Span embedder')
               
        # If span end indices are exclusive, subtract 1
        if self.span_end_is_exclusive:            
            starts, ends = span_indices.split(1, dim=-1)        
            span_indices = torch.cat((starts, ends-1), dim=-1)

        # Initialize output
        X_span = []
        
        # Endpoint embedding
        if self.use_endpoint:
            X_endpoint = self._endpoint_extractor( \
                                    sequence_tensor = sequence_tensor,
                                    span_indices = span_indices,
                                    sequence_mask = sequence_mask,
                                    span_indices_mask = span_mask)
            X_span.append(X_endpoint)
            
            if verbose:
                logging.info('Endpoint extractor used')
                logging.info('Endpoint embedding:\t{}'.format(X_endpoint.shape))
        
        # Attentive embedding
        if self.use_attention:
            X_attn = self._attentive_extractor( \
                                    sequence_tensor = sequence_tensor,
                                    span_indices = span_indices,
                                    sequence_mask = sequence_mask,
                                    span_indices_mask = span_mask)
            X_span.append(X_attn)

            if verbose:
                logging.info('Attentive extractor used')
                logging.info('Attention embedding:\t{}'.format(X_attn.shape))


        # Heuristic positional embedding
        if self.use_position_heuristic:

            #if self.position_embed.device != span_indices.device:
            #    self.position_embed = self.position_embed.to(span_indices.device)

            # Get span midpoints
            midpoints = torch.round(torch.mean(span_indices.float(), dim=-1)).long()
            
            # Get positional embeddings
            X_pos = self.position_embed(midpoints)
            X_span.append(X_pos)

            if verbose:
                logging.info('Position heuristic used')
                logging.info('Position midpoints:\t{}'.format(midpoints.shape))
                logging.info('Position embedding:\t{}'.format(X_pos.shape))

        # Learned positional embedding
        if self.use_position_learned:
            raise ValueError("Model currently does not implement learned positional embeddings")
        
        # Concatenate embeddings
        X_span = torch.cat(X_span, dim=2)
        if verbose:
            logging.info('Concatenated embedding:\t{}'.format(X_span.shape))
        
        # Project embedding
        if self.project:            
            X_span = self.ffnn(X_span)
            if verbose:
                logging.info('Projection')
                logging.info('Projected embedding:\t{}'.format(X_span.shape))


        return X_span

#def get_device_of(tensor: torch.Tensor):
#    """
#    Returns the device of the tensor.
#    """
#    if not tensor.is_cuda:
#        return -1
#    else:
#        return tensor.get_device()
        


#def get_range_vector(size, device):
#    """
#    Returns a range vector with the desired size, starting at 0. The CUDA implementation
#    is meant to avoid copy data from CPU to GPU.
#    """
#    if device == 'cuda':
#        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
#    elif device == 'cpu':
#        return torch.arange(0, size, dtype=torch.long)
#    elif device == None:
#        return torch.arange(0, size, dtype=torch.long)
#    else:
#        raise ValueError("Could not resolve device:\t{}".format(device))

def positional_features(timesteps, hidden_dim, device=None, \
                                min_timescale=1.0, max_timescale=1.0e4):

    """
    Implements the frequency-based positional encoding described
    in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).
    Adds sinusoids of different frequencies to a `Tensor`. A sinusoid of a
    different frequency and phase is added to each dimension of the input `Tensor`.
    This allows the attention heads to use absolute and relative positions.
    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.
    # Parameters
    tensor : `torch.Tensor`
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : `float`, optional (default = 1.0)
        The smallest timescale to use.
    max_timescale : `float`, optional (default = 1.0e4)
        The largest timescale to use.
    # Returns
    The input tensor augmented with the sinusoidal frequencies.
    """  # noqa


    #timestep_range = get_range_vector(timesteps, device).data.float()
    timestep_range = get_range_vector(timesteps, device).data.float()
    
    
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, device).data.float()

    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return sinusoids

def positional_feature_plot( \
                path='positional_feat_plot.png', 
                timesteps=30, 
                hidden_dim=20, 
                device='cpu', 
                min_timescale=1.0, 
                max_timescale=100):

    y = positional_features(timesteps=timesteps, \
                            hidden_dim=hidden_dim, 
                            device=device, 
                            min_timescale=min_timescale, 
                            max_timescale=max_timescale)
                            
    print('y.shape', y.shape)                            
    #y = y.squeeze()    
    z = y.numpy()
    
    # Save figure
    #fig = ax.get_figure()
    z = np.transpose(z)
    ax = sns.heatmap(z, linewidth=0.5)
    ax.set_xlabel('seq')
    ax.set_ylabel('dim')
    fig = ax.get_figure()
    fig.savefig(path, quality=95)    

def add_positional_features(tensor, min_timescale=1.0, max_timescale=1.0e4):

    """
    Implements the frequency-based positional encoding described
    in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).
    Adds sinusoids of different frequencies to a `Tensor`. A sinusoid of a
    different frequency and phase is added to each dimension of the input `Tensor`.
    This allows the attention heads to use absolute and relative positions.
    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.
    # Parameters
    tensor : `torch.Tensor`
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : `float`, optional (default = 1.0)
        The smallest timescale to use.
    max_timescale : `float`, optional (default = 1.0e4)
        The largest timescale to use.
    # Returns
    The input tensor augmented with the sinusoidal frequencies.
    """  # noqa
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.float()

    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)















def tensor_agg(x, agg, dim=-1):
    
    if agg == 'mean':
        y = x.mean(dim)
    elif agg == 'sum':
        y = x.sum(dim)    
    elif agg == 'max':
        y, _ = x.max(dim)
    elif agg == 'min':
        y, _ = x.min(dim)    
    
    return y


def span_embed_agg(X, spans, agg='mean', end_is_exclusive=True, 
    verbose=False):
    '''
    Calculate span embedding from sequence and bedding, 
    by summing across the sequence dimension
    
    Parameters
    ----------
    X: sequence embedding (batch_size, seq_len, embed_dim)
    spans: (batch_size, num_spans, 2)
    end_is_exclusive: Boolean indicating whether the last (end) 
            span indices is exclusive, similar to Python, C++, etc.
    
    '''


    # If and is exclusive, need to decrement, 
    # for compliance with AllenNLP
    if end_is_exclusive:
        starts, ends = spans.split(1, dim=-1)        
        spans = torch.cat((starts, ends-1), dim=-1)

        if verbose:
            logging.info('\tSpan and indices is exclusive, so decrement for AllenNLP')

    # Extract spans from embedding
    # span_select: (batch_size, num_spans, max_span_len, embed_dim)
    # span_mask: (batch_size, num_spans, max_span_len)
    span_select, span_mask = batched_span_select(X, spans)

    # Apply mask
    # (batch_size, num_spans, max_span_len, embed_dim)
    span_select_masked = span_select*(span_mask.unsqueeze(-1))

    dim = 2

    if verbose:
        logging.info('Span embed agg')

    if isinstance(agg, list):
        span_embed = []
        for a in agg:
            x = tensor_agg(span_select_masked, agg=a, dim=dim)
            span_embed.append(x)
            
        span_embed = torch.cat(span_embed, dim=dim)
    else:
        span_embed = tensor_agg(span_select_masked, agg=agg, dim=dim)

    if verbose:
        logging.info('\tX:\t{}'.format(X.shape))
        logging.info('\tspans:\t{}'.format(spans.shape))
        logging.info('\tagg:\t{}'.format(agg))        
        logging.info('\tspan_mask:\t{}'.format(span_mask.shape))
        logging.info('\tspan_select:\t{}'.format(span_select.shape))
        logging.info('\tspan_select_masked:\t{}'.format(span_select_masked.shape))
        logging.info('\tspan_embed:\t{}'.format(span_embed.shape))


    return span_embed
    