
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict, Counter
import logging
import numpy as np
import pandas as pd
import copy

# from models.multitask.dataset import preprocess_X 
from models.multitask.dataset import preprocess_X_xfmr as preprocess_X 

# __NIC_CHANGE__ 
# Looks like change from: 
#   https://github.com/Lybarger/clinical_extraction/commit/89789fdd00512c342c9c74d0b56027745ab56e7b#diff-2540aeb536f0160284ed24caf39fd81a


from corpus.event import Event, Span, events2sent_labs, events2seq_tags
from models.utils import pad_embedding_seq, trunc_seqs, map_dict_builder, pad1D, pad2D, create_mask, one_hot, \
                         get_predictions, get_argument_pred, create_mask, map_dict_builder
from corpus.labels import get_span_label
from corpus.event import Event, Span
from constants import *
from models.xfmrs import tokens2wordpiece, get_embeddings, embed_len_check
from utils.misc import dict_to_list
from models.span_scoring import filter_overlapping
from sklearn.preprocessing import normalize
from itertools import combinations_with_replacement as combos_with_replace, product

from corpus.event import is_overlap

DUMMY_INDICES = (-1, -1)


def len_check(X, Y):
    assert len(X) == len(Y), '{} vs {}'.format(len(X), len(Y))

def len_check_nest(X, Y):
    len_check(X, Y)
    for x, y in zip(X, Y):
        len_check(x, y)
    

def span_maps(label_def):
    '''
    Get mapping dictionaries from list of labels
    '''
    
    # Get mapping from label to ID
    to_id = OrderedDict()
    to_lab = OrderedDict()
    all_keys = set([])
    for name, lab_def in label_def.items():

        # Unpack dictionary
        event_types = lab_def[EVENT_TYPE]
        argument = lab_def[ARGUMENT]
        label_map = lab_def[LAB_MAP]
        label_type = lab_def[LAB_TYPE]                

        to_id[name] = OrderedDict() 
        to_lab[name] = OrderedDict() 
        
        # Map string labels to IDs
        if label_type == EVENT_TYPE:    
            assert isinstance(argument, str)
            for i, lab in enumerate(label_map):
                k = (lab, argument, lab)
                assert k not in all_keys, "{} in {}".format(k, all_keys)
                all_keys.add(k)
                to_id[name][k] = i
                
            
        elif label_type == ARGUMENT:
            assert isinstance(argument, list)
            assert argument == label_map[1:]
            for evt_typ in event_types:               
                for i, lab in enumerate(label_map):
                    k = (evt_typ, lab, None)
                    if lab != OUTSIDE:
                        assert k not in all_keys, "{} in {}".format(k, all_keys)
                    all_keys.add(k)
                    to_id[name][k] = i
        
        elif label_type == LABEL:
            assert isinstance(argument, str)
            for evt_typ in event_types:
                for i, lab in enumerate(label_map):
                    k = (evt_typ, argument, lab)
                    assert k not in all_keys, "{} in {}".format(k, all_keys)
                    all_keys.add(k)
                    to_id[name][k] = i
        
        else:
            raise ValueError("invalid label type:\t{}".format(label_type))
        
        # Map label ID to string labels
        for i, lab in enumerate(label_map):
            to_lab[name][i] = lab

    logging.info('-'*72)
    logging.info('Label-ID mapping:')
    logging.info('-'*72)
    logging.info('Label to ID map:')
    for name, map_ in to_id.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))
                
    logging.info('ID to Label map:')
    for name, map_ in to_lab.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))
    
    return (to_id, to_lab)


def seq_maps(span_id_to_labels):
    
    span_to_seq = OrderedDict()
    seq_to_span = OrderedDict()
    for name, map_ in span_id_to_labels.items():
        
        n = len(map_)
        n_pos = n - 1

        spn2seq = OrderedDict()
        seq2spn = OrderedDict()
        for id_, lab in map_.items():

            if id_ == 0:
                spn2seq[id_] = (id_, id_)
            else:
                spn2seq[id_] = (id_, id_ + n_pos)


            seq2spn[id_] = id_
            if id_ != 0:
                seq2spn[id_ + n_pos] = id_
        



        span_to_seq[name] = spn2seq
        seq_to_span[name] = seq2spn
        
        
        
    constraints = OrderedDict()
    exclusions = OrderedDict()
    for name in seq_to_span:
        
        spn2seq = span_to_seq[name]
        seq2spn = seq_to_span[name]
        

        n = len(spn2seq)
        n_pos = n - 1
               

        all_perm = list(product(list(seq2spn.keys()), repeat=2))
        
        
        
        BI_trans = set([fs for _, fs in spn2seq.items() if fs != (0,0)])
        Bs = set([f for _, (f, s) in spn2seq.items() if f != 0])
        
        c = []
        e = []
        for f, s in all_perm:
        
            # No change in state
            no_change = f == s
        
            # Transition from BEGIN
            to_begin = s in Bs
            
            # Transition to OUTSIDE
            to_outside = s == 0
            
            # Transition from BEGIN to INSIDE
            begin_to_inside = (f, s) in BI_trans
        
        
            if no_change or to_begin or to_outside or \
                  begin_to_inside:
                c.append((f, s))    
            else:
                e.append((f, s))            

        constraints[name] = c
        exclusions[name] = e


    tag_to_span_fn = OrderedDict()
    for name, spn2seq in span_to_seq.items():

        map_ = {}
        for lab, (f, s) in spn2seq.items():
            if f == 0:
                #         (tag, is_O,  is_B,  is_I)
                map_[f] = (f,   True,  False, False)
            else:
                #         (tag, is_O,  is_B,  is_I)
                map_[f] = (f,   False, True,  False)
                
                #         (tag, is_O,  is_B,  is_I)
                map_[s] = (f,   False, False, True)                
                            
        def tag2span(tag, map_=map_):
            return map_[tag]
    
        tag_to_span_fn[name] = tag2span

    num_tags = OrderedDict()
    for name, seq2spn in seq_to_span.items():
        num_tags[name] = len(seq2spn)

    logging.info('-'*72)
    logging.info('Sequence tag-ID mapping')
    logging.info('-'*72)

    logging.info('')
    logging.info('Span to Sequence map:')
    for name, map_ in span_to_seq.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Sequence to Span map:')
    for name, map_ in seq_to_span.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Num tags:')
    for name, n in num_tags.items():
        logging.info('{} = {}'.format(name, n))


    logging.info('')    
    logging.info('-'*72)
    logging.info('Constraints:')
    logging.info('-'*72)
    for name, K in constraints.items():
        logging.info('')
        logging.info(name)
        logging.info('{}'.format(K))

    logging.info('')    
    logging.info('-'*72)
    logging.info('Exclusions:')
    logging.info('-'*72)
    for name, K in exclusions.items():
        logging.info('')
        logging.info(name)
        logging.info('{}'.format(K))

    logging.info('-'*72)
    logging.info('Space reduction:')
    logging.info('-'*72)
    for k in constraints:
        logging.info(k)
        c = len(constraints[k])
        e = len(exclusions[k])
        t = c + e
        r = float(c)/float(t)
        sr = r*r
        logging.info('\tPermutation length:\t{}'.format(t))
        logging.info('\tConstraint length:\t{}'.format(c))
        logging.info('\tExclusions length:\t{}'.format(e))
        logging.info('\tLinear ratio:\t\t{:.2f} ({}/{})'.format(r, c, t))
        logging.info('\tSquared ratio:\t\t{:.2f}'.format(sr))

        
    return (span_to_seq, seq_to_span, tag_to_span_fn, num_tags, constraints)
    

def get_num_tags(label_def):
    '''
    Get tag count
    '''
    
    num_tags = OrderedDict()
    for name, lab_def in label_def.items():
        num_tags[name] = len(lab_def[LAB_MAP])

    logging.info('')
    logging.info('Number of tags:')
    for name, cnt in num_tags.items():
        logging.info('{} = {}'.format(name, cnt))

    return num_tags


def map_1D(X, map_dict):
    '''
    Apply map to iterable (e.g. list, numpy array, etc.)
    '''
    if isinstance(map_dict, dict):
        return [map_dict[x] for x in X]
    else:
        return [map_dict(x) for x in X]
    
def map_2D(X, map_dict):
    return [map_1D(x, map_dict) for x in X]


def mask_2D(X, Y):
    '''
    Build 2D mask from 2 1D vectors
    '''    
    X = X.unsqueeze(1).repeat(1, Y.shape[0])
    Y = Y.unsqueeze(0).repeat(X.shape[0], 1)
    return X*Y
    

def import_idx(indices, pad_start=True):

    # Return dummy value if indices None
    if indices is None:
        return DUMMY_INDICES
    
    return tuple(np.array(indices) + int(pad_start))

    
def export_idx(indices, pad_start=False):
    
    # Return dummy value if dummy value
    if tuple(indices) == DUMMY_INDICES:
        return DUMMY_INDICES 
    
    # Separate start and end indices
    start, end = tuple(indices)
    
    # Account for start of sequence padding
    start = start - int(pad_start)
    end = end - int(pad_start)
    
    return (start, end)    


def get_span_overlaps(span_indices):

    n = len(span_indices)
    span_overlaps = np.zeros((n, n))
    for i1, idxs1 in enumerate(span_indices):
        for i2, idxs2 in enumerate(span_indices):
            span_overlaps[i1, i2] = is_overlap(idxs1, idxs2)

    return span_overlaps



def enumerate_spans(seq_length, \
            ignore_first = True, 
            ignore_last = True,
            max_span_width = None, 
            min_span_width = 1):   
                
    '''
    Enumerate all possible spans
    
    Used code and approach from AllenNLP    
        https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py
    
    NOTE: span end indices are EXCLUSIVE
    '''

    logging.info('='*72)
    logging.info('Enumerating spans')
    logging.info('='*72)
        
    # Get maximum span with if not provided
    if max_span_width is None:
        max_span_width = seq_length
      
    # Start position
    if ignore_first:
        start = 1 
        logging.info('Excluding spans with token position 0')
    else:
        start = 0
        logging.info('Including spans with token position 0')
    
    # Adjust sequence length
    if ignore_last:
        logging.info('Excluding spans with last token position')
    else:
        logging.info('Including spans with last token position')
    seq_length = seq_length - int(ignore_last)

    # Iterate over token positions from start through max length    
    span_indices = []
    for i in range(start, seq_length):
    
        # End indices to traverse
        first_end_index = min(i + min_span_width - 1, seq_length)
        last_end_index = min(i + max_span_width, seq_length)
                    
        # Loop on start-end combinations
        for j in range(first_end_index, last_end_index):
            span_indices.append((i, j + 1))


    # Number of spans for maximum sentence length
    num_spans = len(span_indices)
        
    # Mapping from span token indices to span index
    span_map = OrderedDict((s, i) for i, s in enumerate(span_indices))
  
    # Create span mask for all possible sequence lengths
    span_mask = OrderedDict()
    for seq_len in range(seq_length + 2):
        span_mask[seq_len] = [int(i <= seq_len) for _, i in span_indices]

    # Determine which spans overlap
    span_overlaps = get_span_overlaps(span_indices)
          
    cnt = 15
    
    logging.info('Total span count:\t{}'.format(num_spans))
    logging.info('Minimum span index:\t{}'.format(min([s for s, e in span_indices])))
    logging.info('Maximum span index:\t{}'.format(max([e for s, e in span_indices])))
    logging.info('Span examples:\n{}{}{}'.format(span_indices[0:cnt], '.'*10, span_indices[-cnt:]))
    logging.info('Span map:')
    for j, (s, i) in enumerate(span_map.items()):
        if (j < cnt) or (num_spans - j < cnt):
            logging.info('\t{} --> {}'.format(s, i))
    logging.info('Span mask:')
    for j, (len_, msk) in enumerate(span_mask.items()):
        if (j < cnt) or (len(span_mask) - j < cnt):
            logging.info('\tseq len={} --> mask={}'.format(len_, msk))
    logging.info('Span overlaps:\n{}'.format(span_overlaps))

    
    return (span_indices, span_map, span_mask, num_spans, span_overlaps)
    


def get_indicators(events, label_to_id, max_len, pad_start=True):
    '''
    Get tensor representation for mention (e.g. trigger)
    
    Parameters
    ----------
    
    events: document events as list of list of Event
    span_map: dictionary for mapping span token indices to span index
    
    
    label_map: dictionary for mapping labels to label indices    
    pad_start: boolean indicating whether token indices should be incremented to account for start of sequence padding
    
    Returns
    -------
    
    labels: list of span labels as tuple (span idx, label)
    
    '''
    
    
    
    # Number of tags (labels)
    ids = sorted(list(set([v for k, v in label_to_id.items()])))
    num_tags = len(ids)

    assert isinstance(events, list)
    assert isinstance(events[0], list)
    
    # Sentence count
    n_sent = len(events)

    # Loop on sentences in document
    sent_labels = np.zeros((n_sent, num_tags)) 
    token_scores = np.zeros((n_sent, max_len, num_tags)).astype(float)

    for sent_events in events:
    
        # Loop on events in sentence
        for event in sent_events:

            # Loop on arguments in event
            for i, span in enumerate(event.arguments):

                # First span should be the trigger
                if i == 0:
                    assert span.type_ == TRIGGER
                                
                # Is event-argument applicable to label type
                key = (event.type_, span.type_, span.label)
                key_match = key in label_to_id

                # Include span            
                if key_match:
                    
                    # Label ID
                    lab_id = label_to_id[key]
                
                    # Sentence index of span
                    sent_idx = span.sent_idx
                    assert sent_idx in range(n_sent)
                
                    # Include span label
                    sent_labels[sent_idx, lab_id] = 1
                    
                    start, end = import_idx(span.tok_idxs, pad_start)
                    token_scores[sent_idx, start:end, lab_id] = 1
                    
    # Force null label to be 0, if any positive labels
    sent_labels[:,0] = (sent_labels[:,1:].sum(axis=1) == 0).astype(int)

    # Normalize token_scores    
    for i in range(n_sent):
        for j in range(num_tags):
            N = np.sum(token_scores[i,:,j])
            if N > 0:
                token_scores[i,:,j] = token_scores[i,:,j]/N

    return (sent_labels, token_scores)


def get_seq_labels(span_labels, span_map, span_to_seq, max_len):

    '''
    Get tensor representation for mention (e.g. trigger)
    
    Parameters
    ----------
    
    
    '''
        
    # Create reverse span map, mapping span indices to token indices
    rev_span_map = OrderedDict([(i, s) for s, i in span_map.items()])
    
    # Number of tags (labels)
    num_tags = len(span_to_seq)

    # Sentence count
    n_sent = len(span_labels)

    # Loop on sentences in document   
    seq_labels = np.zeros((n_sent, max_len)).astype(int)

    # Iterate over sequences
    for i, spans in enumerate(span_labels):
        
        # Iterate over spans in sequence
        for span_idx, span_lab in spans:
            
            # Gets token indices
            start, end = rev_span_map[span_idx]
            
            # Get applicable begin/inside labels
            B, I = span_to_seq[span_lab]
            
            # Update sequence labels
            seq_labels[i, start:end] = I
            seq_labels[i, start] = B
            
    return seq_labels






def get_mention_labels(events, span_map, label_to_id, pad_start):
    '''
    Get tensor representation for mention (e.g. trigger)
    
    Parameters
    ----------
    
    events: document events as list of list of Event
    span_map: dictionary for mapping span token indices to span index
    
    
    label_map: dictionary for mapping labels to label indices    
    pad_start: boolean indicating whether token indices should be incremented to account for start of sequence padding
    
    Returns
    -------
    
    labels: list of span labels as tuple (span idx, label)
    
    '''
    
    assert isinstance(events, list)
    assert isinstance(events[0], list)
    
    # Sentence count
    n_sent = len(events)

    # Loop on sentences in document
    labels = [[] for _ in range(n_sent)]
    for sent_events in events:
    
        # Loop on events in sentence
        for event in sent_events:

            # Loop on arguments in event
            for i, span in enumerate(event.arguments):

                # First span should be the trigger
                if i == 0:
                    assert span.type_ == TRIGGER
                                
                # Is event-argument applicable to label type
                key = (event.type_, span.type_, span.label)
                key_match = key in label_to_id

                # Span indices
                tok_idxs = import_idx(span.tok_idxs, pad_start) 
                
                # Are span indices in span map
                span_match = tok_idxs in span_map

                # Include span            
                if key_match and span_match:
                    
                    # Label ID
                    lab_id = label_to_id[key]

                    # Index of span     
                    span_idx = span_map[tok_idxs]
                
                    # Sentence index of span
                    sent_idx = span.sent_idx
                    assert sent_idx in range(n_sent)
                
                    # Include span label
                    labels[sent_idx].append((span_idx, lab_id))
    
    return labels


def get_mention_tensor(labels, num_spans):
    '''
    Convert label mentions two padded tensor
    
    Parameters
    ----------
    labels: list of tuple (span_index, span_label)
    num_spans: int of number of enumerated spans
    
    Returns
    -------
    tensor: pytorch tensor with shape (num_spans)
    '''
    # Initialize
    tensor = torch.zeros(num_spans, dtype=torch.long)
    
    # Loop on labels
    for idx, lab in labels:
        tensor[idx] = lab

    return tensor


def get_arg_labels(events, span_map, trig_map, arg_map, pad_start):

    '''
    Get argument tensors
    
    
    Parameters
    ----------
    
    events: document of events, list of list of Event
    arg_type: string indicating argument type (e.g.'Trigger' or 'Status')
    num_trig: int indicating maximum number of trigger span_indices
    num_arg: int indicating maximum number of argument spans
    arg_map: dictionary for mapping argument labels to indices
    span_map: dictionary for mapping span labels to indices
    label_type: string indicating how to represent argument label (e.g. event type, span type, etc.)
    
    
    Returns
    -------
    
    arg_labels: tensor of trigger-argument labels with shape (num_trig, num_arg)
    mask: tensor of trigger-argument integer mask with shape (num_trig, num_arg)
    span_labels: tensor of argument labels with shape (num_arg)
    span_indices: tensor of mention token indices with shape (num_arg, 2)
            NOTE that span indices are Inclusive for Allen NLP
    
    '''

    assert isinstance(events, list)
    assert isinstance(events[0], list)

    # Sentence count
    n_sent = len(events)
    
    # Loop on sentences in document
    labels = [[] for _ in range(n_sent)]
    for sent_events in events:
    
        # Loop on events in sentence
        for event in sent_events:
            
            # Get arguments
            arguments = copy.deepcopy(event.arguments)
            
            # Get arguments
            trig = arguments.pop(0)
            assert trig.type_ == TRIGGER
            
            # Trigger key, token indices, and sentence index
            trig_key = (event.type_, trig.type_, trig.label)
            trig_tok_idx = import_idx(trig.tok_idxs, pad_start)
            trig_sent_idx = trig.sent_idx
            
            # Iterate over remaining arguments
            for arg in arguments:
            
                # Argument key, token indices, and sentence index
                arg_key = (event.type_, arg.type_, arg.label)
                arg_tok_idx = import_idx(arg.tok_idxs, pad_start)
                arg_sent_idx = arg.sent_idx
            
                # Trigger an argument in same sentence
                sent_match = trig_sent_idx == arg_sent_idx
                
                # Trigger and argument applicable to current map
                key_match = (trig_key in trig_map) and \
                            (arg_key in arg_map)
                
                # Trigger and argument spans in span map
                span_match = (trig_tok_idx in span_map) and \
                             (arg_tok_idx in span_map)
                
                # Include argument
                if sent_match and key_match and span_match:

                    # Trigger and span indices
                    i_trig = span_map[trig_tok_idx]
                    i_arg = span_map[arg_tok_idx]
                    
                    assert trig_sent_idx in range(n_sent)

                    # Save label triple
                    labels[trig_sent_idx].append((i_trig, i_arg, 1))
    return labels


def get_arg_tensor(labels, num_spans):
    '''
    Convert label mentions two padded tensor
    
    Parameters
    ----------
    labels: list of tuple (trigger_index, arg_index, span_label)
    num_spans: int of number of enumerated spans
    '''
    # Initialize
    tensor = torch.zeros(num_spans, num_spans, dtype=torch.long)
    
    # Loop on labels
    for trig_idx, arg_idx, lab in labels:
        tensor[trig_idx][arg_idx] = lab

    return tensor



def preprocess_events(events, label_to_id, span_to_seq, span_map, pad_start, max_len):
    '''
    Extract labels from document of events


    Parameters
    ----------
    
    events: list of list of Event (document of Events)
   
    
    Returns
    -------
    d: dictionary of tensors:
            trig_labels (num_trig)
            trig_mask (num_trig)
            trig_spans (num_trig, 2)
            status_labels (num_status)
            status_spans (num_status, 2)
            entity_labels (num_entity)
            entity_spans (num_entity, 2)                
            status_arg_labels (num_trig, num_status)
            status_arg_mask (num_trig, num_status)

            entity_arg_labels (num_trig, num_entity) (binary label)
            entity_arg_mask (num_trig, num_entity)
                    
    '''

    # Sentence count
    sent_count = len(events)
  
    # Iterate over label types
    mention_labels = OrderedDict()
    indicator_labels = OrderedDict()
    indicator_weights = OrderedDict()
    seq_labels = OrderedDict()    
    arg_labels = OrderedDict()
    for name, lab_to_id in label_to_id.items():
        
        # Mention labels
        mention_labels[name] = get_mention_labels( \
                                events = events, 
                                span_map = span_map, 
                                label_to_id = lab_to_id,
                                pad_start = pad_start)

        indicator_labels[name], indicator_weights[name] = \
                                     get_indicators(events = events, 
                                                label_to_id = lab_to_id,
                                                max_len = max_len, 
                                                pad_start = pad_start)
        
        seq_labels[name] = get_seq_labels(mention_labels[name], 
                                      span_map = span_map,
                                      span_to_seq = span_to_seq[name],
                                      max_len = max_len)
        
        # Argument labels
        if name != TRIGGER:
            arg_labels[name] = get_arg_labels( \
                            events = events, 
                            span_map = span_map, 
                            trig_map = label_to_id[TRIGGER],
                            arg_map = label_to_id[name], 
                            pad_start = pad_start)

    
            assert len(mention_labels[name]) == len(arg_labels[name])

    def lc(k, x, y):
        assert x == y, "{}: {} vs {}".format(k, x, y)

    # Check sentence counts
    for k, v in mention_labels.items():
        lc(k, len(v), sent_count)
    for k, v in indicator_labels.items():
        lc(k, len(v), sent_count)        
    for k, v in indicator_weights.items():
        lc(k, len(v), sent_count)        
    for k, v in seq_labels.items():
        lc(k, len(v), sent_count)
    for k, v in arg_labels.items():
        lc(k, len(v), sent_count)

    # Convert to list of dictionary
    labels = []
    for i in range(sent_count):
        ml = OrderedDict()
        il = OrderedDict()
        iw = OrderedDict()        
        sl = OrderedDict()
        al_ = OrderedDict()
    
        for k, v in mention_labels.items():
            ml[k] = v[i]

        for k, v in indicator_labels.items():
            il[k] = v[i]

        for k, v in indicator_weights.items():
            iw[k] = v[i]

        for k, v in seq_labels.items():
            sl[k] = v[i]
            
        for k, v in arg_labels.items():
            al_[k] = v[i]
            
        d = {'mention_labels': ml, \
             'indicator_labels': il, 
             'indicator_weights': iw,
             'seq_labels': sl,                  
             'arg_labels':     al_}        
        labels.append(d)
        
    return labels


def build_span(mention_id, label_type, arg_type, id_to_lab, span, pad_start):
    '''
    
    Parameters
    ----------
    mention_id: predicted label ID for mention
    label_type: type of label, either EVENT_TYPE, LABEL, or ARGUMENT
    arg_type: argument type, e.g. STATUS, TYPE, etc.
    id_to_lab: map from predicted label id to label
    span: span indices
    pad_start: boolean indicating whether start is padded

    '''
    
    # Map label ID to label
    mention_label = id_to_lab[mention_id]
    
    # Determine span type_ and label
    if label_type == LABEL:
        type_ = arg_type
        label = mention_label
    elif label_type == ARGUMENT:
        type_ = mention_label
        label = None         
    elif label_type == EVENT_TYPE:
        type_ = TRIGGER 
        label = mention_label
    else:
        raise ValueError("Invalid label_type:\t{}".format(label_type))                                          
    
    # Token indices
    tok_idxs = export_idx( \
            indices = span, 
            pad_start = pad_start)
    
    # Build span
    span = Span( \
            type_ = type_, 
            sent_idx = None, 
            tok_idxs = tok_idxs, \
            label = label) 

    return span


def tensor_dict_to_numpy(d):
    return OrderedDict([(name, v.cpu().numpy()) for name, v in d.items()])

def get_ment_arg_pred(mention_scores, mention_spans, mention_mask, \
                        arg_scores, arg_mask, label_def, 
                        prune_overlapping=False, verbose=False):
                        # overlaps=None, 

    # Get mention labels from scores and mask
    mention_labels = OrderedDict()
    for name in mention_scores:
        
        if prune_overlapping:
            mention_scores[name] = filter_overlapping( \
                        logits = mention_scores[name], 
                        mask = mention_mask[name], 
                        spans = mention_spans[name])       
        
        
        mention_labels[name] = get_predictions( \
                                    scores = mention_scores[name], 
                                    mask = mention_mask[name])
    # Get argument predictions
    arg_labels = OrderedDict()
    for name in arg_scores:
        # (batch_size, num_trig, num_arg)
        arg_labels[name] = get_argument_pred( \
                                span_pred = mention_labels[name], 
                                arg_scores = arg_scores[name], 
                                arg_mask = arg_mask[name],
                                is_req = label_def[name][REQUIRED])

    
    if verbose:
        logging.info('\tMention labels:')
        for name, v in mention_labels.items():
            logging.info('\t\t{}:\t{}'.format(name, v.shape))
        logging.info('\tArgument labels:')
        for name, v in arg_labels.items():
            logging.info('\t\t{}:\t{}'.format(name, v.shape))

    # Convert to numpy arrays
    mention_labels  = tensor_dict_to_numpy(mention_labels)
    arg_labels      = tensor_dict_to_numpy(arg_labels)
    
    return (mention_labels, arg_labels)



def X_filt(X, includes_mask):
    
    ls = lambda cnt: [[] for _ in range(cnt)]    
        
    doc_count = len(X)

    # Input sentences include a binary math
    if includes_mask:

        mask = ls(doc_count)
        X_tmp = ls(doc_count)
        
        # Loop on documents
        for i, doc in enumerate(X):
           
            # Loop on sentences
            for j, (msk, toks) in enumerate(doc):
                
                # Include all mask values
                mask[i].append(msk)
                
                # Only include masked X and Y
                if msk:
                    X_tmp[i].append(toks)
                
        
        # Reassign vars        
        X = X_tmp                    

    # If no mask included with X, just need to create dummy mask
    else:
        mask = []
        for doc in X:
            n_sent = len(doc)
            mask.append([1]*n_sent)

    return (X, mask)


def Y_filt(Y, mask):

    if Y is None:
        return Y
    
    #n_Y =    sum([len(y) for y in Y])
    n_Y =    len(Y)
    n_mask = sum([len(y) for y in mask])
    assert n_Y == n_mask, 'n_Y = {} vs. n_mask = {}'.format(n_Y, n_mask)
    

    mask = [m for M in mask for m in M]
    len_check(Y, mask)
    Y_out = [y for y, m in zip(Y, mask) if m]

    # Loop on documents
    #len_check(Y, mask)
    #Y_out = []
    #for y_doc, m_doc in zip(Y, mask):
    #    
    #    # Create space for current doc
    #    Y_out.append([])
    #    
    #    # Loop on sentences in current document
    #    len_check(y_doc, m_doc)
    #    for y_sent, m_sent in zip(y_doc, m_doc):
    #        
    #        # Include current sentence
    #        if m_sent:
    #            Y_out[-1].append(y_sent)
    
    return Y_out
    
    
class EventExtractorDataset(Dataset):
    """
    This dataset contains a list of numbers in the range [a,b] inclusive
    """
    
    
    def __init__(self, X, Y, \
        max_len = 30, 
        label_def = None,         
        use_xfmr = True,
        xfmr_type = None,
        xfmr_dir = None,
        word_embed = None,
        max_span_width = 8, 
        min_span_width = 1,
        num_workers = 6,
        #batch_size = 200,
        device = None,
        X_includes_mask = False,
        ):
        super(EventExtractorDataset, self).__init__()


        logging.info('='*72)
        logging.info('EventExtractorDataset')
        logging.info('='*72)

        self.pad_start = True
        self.pad_end = True

        # Assign input attributes
        self.max_len = max_len
        self.label_def = label_def
        self.use_xfmr = use_xfmr
        self.xfmr_type = xfmr_type
        self.xfmr_dir = xfmr_dir
        self.word_embed = word_embed
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        self.num_workers = num_workers
        #self.batch_size = batch_size
        self.device = device
        self.X_includes_mask = X_includes_mask

        # Get label to ID and ID to label maps
        self.label_to_id, self.id_to_label = span_maps(label_def)        
        
        self.span_to_seq, _, _, _, _  = seq_maps(self.id_to_label)


        # Document count
        self.doc_count = len(X)    
        logging.info('Document count:\t{}'.format(self.doc_count))
        seq_count = sum([len(x) for x in X]) 
        logging.info('Sequence count, before filter:\t{}'.format(seq_count))

        # Sentence count, document indices, and sentence indices
        self.sent_counts = []
        for i, doc in enumerate(X):
            n = len(doc)
            self.sent_counts.append(n)


        # Preprocess labels (i.e. convert events to span labels)
        if Y is None:
            Y = [[[] for sent in doc] for doc in X]

   
        X, self.sent_mask = X_filt(X, self.X_includes_mask)
        
        

            
        self.seq_count = sum([len(x) for x in X])
        logging.info('Sequence count, after filter:\t{}'.format(self.seq_count))
    


        # Preprocess input (convert tokens to contextualized embeddings)
        # __NIC_CHANGE__
        self.tokens, X_, self.seq_lengths = preprocess_X( \
                                X = X, 
                                xfmr_type = self.xfmr_type,
                                xfmr_dir = self.xfmr_dir,
                                # use_xfmr = self.use_xfmr,
                                max_len = self.max_len, 
                                # pad_start = self.pad_start, 
                                # pad_end = self.pad_end,
                                num_workers = self.num_workers, 
                                device = self.device)

        assert len(X_) == self.seq_count        
        self.mask = create_mask(self.seq_lengths, self.max_len)   

        self.X = X_ 
        
        # Get all spans
        # Start indices are inclusive
        # End indices are exclusive
        self.span_indices, self.span_map, self.span_mask, self.num_spans, self.span_overlaps = \
                enumerate_spans(self.max_len, \
                                 ignore_first = self.pad_start, 
                                 ignore_last = self.pad_end, 
                                 max_span_width = self.max_span_width, 
                                 min_span_width = self.min_span_width)


        

        self.Y = self.preprocess_Y(Y, self.sent_mask)
        #self.Y = Y_filt(Y, self.sent_mask)
        #self.Y = self.tensorfy_Y(Y_labs)
        #self.Y = Y_filt(self.Y, self.sent_mask)        
        #self.label_dist()        



                    
    def __len__(self):
        return self.seq_count
        
    def __getitem__(self, index):
        X = self.tensorfy_X(self.X[index])     
          
        # X = self.X[index]
        Y = self.tensorfy_Y(self.Y[index], self.seq_lengths[index])        
        #Y = self.Y[index]
        mask = self.mask[index]
        return (X, Y, mask)

        
        
    def preprocess_Y(self, Y, sent_mask=None):
        '''
        Extract labels from events


        Parameters
        ----------
        
        events: list of list of Event
       
        
        Returns
        -------
        labels: list of dictionary of tensors:
                trig_labels (num_trig)
                trig_mask (num_trig)
                trig_spans (num_trig, 2)
                status_labels (num_status)
                status_spans (num_status, 2)
                entity_labels (num_entity)
                entity_spans (num_entity, 2)                
                status_arg_labels (num_trig, num_status)
                status_arg_mask (num_trig, num_status)

                entity_arg_labels (num_trig, num_entity) (binary label)
                entity_arg_mask (num_trig, num_entity)
                        
        '''

        if sent_mask is None:
            sent_mask = [[1 for y_ in y] for y in Y]


        logging.info('='*72)
        logging.info('Preprocessing Y')
        logging.info('='*72)

        # Iterate over documents
        labels = [] 
        len_check(Y, sent_mask)
        for events, msk in zip(Y, sent_mask):
        

        
            # Get labels from events
            labs = preprocess_events( \
                        events = events, 
                        label_to_id = self.label_to_id, 
                        span_to_seq = self.span_to_seq,
                        span_map = self.span_map, 
                        pad_start = self.pad_start,
                        max_len = self.max_len)

            len_check(events, msk)            
            len_check(labs, msk)
            
            for lb, m in zip(labs, msk):
                if m:
                    labels.append(lb)

        return labels


    def tensorfy_X(self, x):
        return torch.tensor(pad_embedding_seq(x, self.max_len))

    def tensorfy_Y(self, d, seq_len):
        
        
       
        #assert len(labels) == len(self.seq_lengths)
        #tensors = []
        #for d, seq_len in zip(labels, self.seq_lengths):
        

        g = OrderedDict()
        
        # Span indices as tensor
        # (num_spans, 2)
        g['span_indices'] = torch.LongTensor(self.span_indices)
        #g['span_midpoint'] = torch.round(torch.mean(torch.FloatTensor(self.span_indices), dim=-1)
        g['span_overlaps'] = torch.BoolTensor(self.span_overlaps)
        g['span_mask'] = torch.LongTensor(self.span_mask[seq_len])
        g['seq_length'] =  torch.LongTensor([seq_len]).squeeze()
        
        
       
        mention_labels = OrderedDict()
        for name, labels in d['mention_labels'].items():
            mention_labels[name] = get_mention_tensor(labels, self.num_spans)

        indicator_labels = OrderedDict()
        for name, labels in d['indicator_labels'].items():
            indicator_labels[name] = torch.LongTensor(labels)

        indicator_weights = OrderedDict()
        for name, labels in d['indicator_weights'].items():
            indicator_weights[name] = torch.FloatTensor(labels)


        seq_labels = OrderedDict()
        for name, labels in d['seq_labels'].items():
            seq_labels[name] = torch.LongTensor(labels)


        arg_labels = OrderedDict()
        for name, labels in d['arg_labels'].items():
            arg_labels[name] = get_arg_tensor(labels, self.num_spans)
        
        g['mention_labels'] = mention_labels
        g['indicator_labels'] = indicator_labels
        g['indicator_weights'] = indicator_weights
        g['seq_labels'] = seq_labels
        g['arg_labels'] = arg_labels
        
        return g    
        #    tensors.append(g)
        
        #return tensors

        

        
        return g        


    def label_dist(self):
        
        assert len(self.seq_lengths) == len(self.Y)
        
        mention_counter = Counter()
        arg_counter = Counter()
        for seq_len, y in zip(self.seq_lengths, self.Y):
            
            yt = self.tensorfy_Y(y, seq_len)
            
            
            for k in yt['mention_labels']:
                labs = yt['mention_labels'][k].numpy()
                #mask = yt['span_mask'][k].numpy()
                mask = yt['span_mask'].numpy()
                labs = (labs*mask).flatten()
                labs = labs[labs != 0]
                for lb_id in labs:
                    lb = self.id_to_label[k][lb_id]
                    mention_counter[(k, lb)] += 1
                    


            for k in yt['arg_labels']:
                labs = yt['arg_labels'][k].numpy()
                #mask = yt['arg_mask'][k].numpy()
                #mask = yt['arg_mask'].numpy()
                mask = mask_2D(yt['span_mask'], yt['span_mask']).numpy()
                
                
                labs = (labs*mask).flatten()
                labs = labs[labs != 0]
                for lb_id in labs:
                    lb = np.asscalar(lb_id)
                    arg_counter[('arg', k, lb)] += 1                

        df = pd.DataFrame.from_dict(mention_counter, orient='index').reset_index()    
        df = df.rename(columns={'index':'event', 0:'count'})
        logging.info('Mention counts:\n{}'.format(df))

        df = pd.DataFrame.from_dict(arg_counter, orient='index').reset_index()    
        df = df.rename(columns={'index':'event', 0:'count'})
        logging.info('Arguent counts:\n{}'.format(df))
            
        return df
            

    def decode_spans(self, scores, spans, mask, \
                 prune_overlapping=False, verbose=False):

        '''
        Decode predictions, converting tensors to list of spans 
        for each sentence
        
        Parameters
        ----------
        scores: dictionary of span scores
        
        trig_pred: tensor of predictions (batch_size, num_trig)
        status_arg_pred: tensor of predictions (batch_size, num_trig, num_status)
        entity_arg_pred: tensor of predictions (batch_size, num_trig, num_status)  
        
        Returns
        -------
        predictions = list of list of spans, e.g. sequences (sentences) of spans
        
        '''

        if verbose:
            logging.info('')
            logging.info('-'*72)
            logging.info('Decode spans: ')
            logging.info('-'*72)


        if prune_overlapping:
            for name in scores:
                scores[name] = filter_overlapping( \
                                            logits = scores[name], 
                                            mask = mask[name], 
                                            spans = spans[name])  
        
        
        batch_size, _, _ = scores[list(scores.keys())[0]].shape
        
        # Iterate over label types (e.g. trigger, status, etc.)
        spans_by_sent = [[] for _ in range(batch_size)]
        for name in scores:

            # Predictions and span indices
            # (batch_size, num_spans, num_tag)
            scores_batch = scores[name]
            # (batch_size, num_spans)
            mask_batch = mask[name]
            labs_batch = get_predictions(scores_batch, mask_batch).cpu().numpy()

            # (batch_size, num_spans, 2)
            spans_batch = spans[name].cpu().numpy()

            if verbose:
                logging.info('\t{}'.format(name))
                logging.info('\t\tscores tensor:\t{}'.format(scores_batch.shape))
                logging.info('\t\tmask tensor:\t{}'.format(mask_batch.shape))
                logging.info('\t\tlabs array:\t{}'.format(labs_batch.shape))                      
                logging.info('\t\tspans array:\t{}'.format(spans_batch.shape))

            # Iterate over sequences in batch
            for i, (labs_seq, spans_seq) in enumerate(zip(labs_batch, spans_batch)):

                # Iterate over spans in sequence
                for lab, spn in zip(labs_seq, spans_seq):

                    # Positive label (non-null label)
                    if lab > 0:
                        
                        span_obj = build_span( \
                                        mention_id = lab, 
                                        label_type = self.label_def[name][LAB_TYPE], 
                                        arg_type = self.label_def[name][ARGUMENT], 
                                        id_to_lab = self.id_to_label[name], 
                                        span = spn, 
                                        pad_start = self.pad_start)
                        
                        # Append to last sequence
                        spans_by_sent[i].append(span_obj)

     
        return spans_by_sent


    def decode_events(self, mention_scores, mention_spans, mention_mask, \
                            arg_scores, arg_mask, prune_overlapping=False, verbose=False):
                                #  overlaps=None,


        '''
        Decode predictions, converting tensors to list of Event 
        for each sentence
        
        Parameters
        ----------
        trig_pred: tensor of predictions (batch_size, num_trig)
        status_arg_pred: tensor of predictions (batch_size, num_trig, num_status)
        entity_arg_pred: tensor of predictions (batch_size, num_trig, num_status)  
        
        Returns
        -------
        predictions = list of sentence events, where each sentence is 
                        represented as a list of Event
        
        '''


        '''
        Predictions
        '''        

        if verbose:
            logging.info('')
            logging.info('-'*72)
            logging.info('Decode events: ')
            logging.info('-'*72)
            
            
            

            
        # Get mention and argument label predictions
        mention_labels, arg_labels = get_ment_arg_pred( \
                                        mention_scores = mention_scores, 
                                        mention_spans = mention_spans, 
                                        mention_mask = mention_mask, 
                                        arg_scores = arg_scores, 
                                        arg_mask = arg_mask, 
                                        label_def = self.label_def,
                                        prune_overlapping = prune_overlapping, 
                                        #overlaps = overlaps,
                                        verbose = verbose)

        mention_spans = tensor_dict_to_numpy(mention_spans)
                
        # Separate trigger and mention_labels
        trig_labels = mention_labels[TRIGGER]
        trig_spans = mention_spans[TRIGGER]

        # Dimensions of trigger
        seq_count, evt_count = trig_labels.shape
        
        if verbose:
            logging.info('\tSequence count:\t{}'.format(seq_count))
            logging.info('\tEvent counts:\t{}'.format(evt_count))
        
        # Loop on sentences in batch
        events = []
        for i in range(seq_count):
            events.append([])

            # Loop on triggers (events)
            for j in range(evt_count):
            
                # Current trigger, scaler
                trig_lab = trig_labels[i][j]
                trig_span = export_idx( \
                                        indices = trig_spans[i][j], 
                                        pad_start = self.pad_start)
                
                # Only decode if trigger predicted
                if trig_lab != 0:

                    '''
                    Trigger span
                    '''
                    trig_lab = self.id_to_label[TRIGGER][trig_lab]

                    # Create event
                    event = Event(type_=trig_lab)

                    # Add trigger
                    trigger = Span( \
                                type_ = TRIGGER, 
                                sent_idx = None, 
                                tok_idxs = trig_span, 
                                label = trig_lab)
                    event.add_argument(trigger)

                    '''
                    Arguments
                    '''
                    
                    # Iterate over argument label groups
                    for name, arg_labs in arg_labels.items():
                    
                        # Arguments for sequence-trigger combination
                        # (num_arg)
                        args = arg_labs[i][j]
                        
                        # Iterate over individual arguments
                        for k, arg in enumerate(args):
                            
                            # Argument predictions are binary
                            # i.e. if arg == 1 then assign to event
                            if (arg == 1) and \
                               (trig_lab in self.label_def[name][EVENT_TYPE]):

                                # Build span from predictions                           
                                span = build_span( \
                                        mention_id = mention_labels[name][i][k], 
                                        label_type = self.label_def[name][LAB_TYPE], 
                                        arg_type = self.label_def[name][ARGUMENT], 
                                        id_to_lab = self.id_to_label[name], 
                                        span = mention_spans[name][i][k], 
                                        pad_start = self.pad_start)
                                 
                                event.add_argument(span)

                    events[-1].append(event)        

        return events                


    def by_doc(self, Y_by_sent):
        '''
        Convert list of sentence representations to document representations
        
        e.g. [sent [Span or Event]] --> [doc [sent [Span or Event]]]
        '''
        
        # Check lengths
        mask_total = sum([sum(x) for x in self.sent_mask])
        assert mask_total == len(Y_by_sent), \
              '{} vs {}'.format(mask_total, len(Y_by_sent))
        assert mask_total == len(self.tokens), \
            '{} vs {}'.format(mask_total, len(self.tokens))

        # Initialize output
        Y_by_doc = [[[] for _ in range(c)] for c in self.sent_counts]        
        
        
        tokens = self.tokens[:]
        
        # Iterate over documents    
        for i, doc_masks in enumerate(self.sent_mask):
            
            # Iterate over sentences in current document
            for j, mask in enumerate(doc_masks):
            
                # Get prediction
                if mask:
                
                    Y = Y_by_sent.pop(0)                    
                    toks = tokens.pop(0)

                    # Iterate over elements in sentence
                    for y in Y:
                    
                        # Processing Span objects
                        if isinstance(y, Span):
                            y.sent_idx = j
                            start, stop = y.tok_idxs
                            y.tokens = toks[start:stop]
                        
                        # Processing Event objects
                        elif isinstance(y, Event):                    
                            for span in y.arguments:
                                span.sent_idx = j
                                start, stop = span.tok_idxs
                                span.tokens = toks[start:stop]
                        else:
                            raise TypeError("incorrect type:\t{}".format(type(y)))                    
                    
                        Y_by_doc[i][j].append(y)

        return Y_by_doc
'''
def seq_maps(span_id_to_labels):
    
    id_to_lab = OrderedDict()
    for name, map_ in span_id_to_labels.items():
        
        n = len(map_)
        n_pos = n - 1

        d = OrderedDict()
        for id_, lab in map_.items():
            d[id_] = lab
        
        for id_, lab in map_.items():
            if id_ != 0:
                d[id_ + n_pos] = lab
        
        id_to_lab[name] = d

    lab_to_id = OrderedDict()
    for name, map_ in span_id_to_labels.items():

        n = len(map_)
        n_pos = n - 1
        
        d = OrderedDict()
        for id_, lab in map_.items():
            if id_ == 0:
                d[lab] = (id_, id_)
            else:
                d[lab] = (id_, id_ + n_pos)
        
        lab_to_id[name] = d


    id_to_id = OrderedDict()
    for name, map_ in span_id_to_labels.items():

        n = len(map_)
        n_pos = n - 1
        
        d = OrderedDict()
        for id_, lab in map_.items():
            d[id_]         = (id_, id_)
            d[id_ + n_pos] = (id_, id_ + n_pos)
        
        id_to_id[name] = d

        
    constraints = OrderedDict()
    exclusions = OrderedDict()
    for name, map_ in id_to_lab.items():

        n = len(lab_to_id[name])
        n_pos = n - 1
               
        # all_combos = list(combos_with_replace(list(map_.keys()), 2))
        all_perm = list(product(list(map_.keys()), repeat=2))
        
        
        
        BI_trans = set([fs for _, fs in lab_to_id[name].items() if fs != (0,0)])
        Bs = set([f for _, (f, s) in lab_to_id[name].items() if f != 0])
        
        c = []
        e = []
        for f, s in all_perm:
        
            # No change in state
            no_change = f == s
        
            # Transition from BEGIN
            to_begin = s in Bs
            
            # Transition to OUTSIDE
            to_outside = s == 0
            
            # Transition from BEGIN to INSIDE
            begin_to_inside = (f, s) in BI_trans
        
        
            if no_change or to_begin or to_outside or \
                  begin_to_inside:
                c.append((f, s))    
            else:
                e.append((f, s))            

        constraints[name] = c
        exclusions[name] = e


    tag_to_span_fn = OrderedDict()
    for name, map_ in lab_to_id.items():
        O = set()
        B = set()
        I = set()
        id_map = {}
        for lab, (f, s) in map_.items():
            if f == 0:
                O.add(f)
            else:
                B.add(f)
                I.add(s)
            id_map[f] = f
            id_map[s] = f
                
        def tag2span(tag, O=O, B=B, I=I, id_map=id_map):
            is_O = tag in O
            is_B = tag in B
            is_I = tag in I            
            lab = id_map[tag]
        
            return (lab, is_O, is_B, is_I)
    
        tag_to_span_fn[name] = tag2span

    num_tags = OrderedDict()
    for name, map_ in id_to_lab.items():
        num_tags[name] = len(map_)

    logging.info('-'*72)
    logging.info('Sequence tag-ID mapping')
    logging.info('-'*72)

    logging.info('')
    logging.info('Label to id map:')
    for name, map_ in lab_to_id.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Id to label map:')
    for name, map_ in id_to_lab.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Num tags:')
    for name, n in num_tags.items():
        logging.info('{} = {}'.format(name, n))

    logging.info('')
    logging.info('Id to Id map:')
    for name, map_ in id_to_id.items():
        logging.info('')
        logging.info(name)
        for k, v in map_.items():
            logging.info('{} --> {}'.format(k, v))



    logging.info('')    
    logging.info('-'*72)
    logging.info('Constraints:')
    logging.info('-'*72)
    for name, K in constraints.items():
        logging.info('')
        logging.info(name)
        for k in K:
            logging.info('{}'.format(k))

    logging.info('')    
    logging.info('-'*72)
    logging.info('Exclusions:')
    logging.info('-'*72)
    for name, K in exclusions.items():
        logging.info('')
        logging.info(name)
        for k in K:
            logging.info('{}'.format(k))

    logging.info('-'*72)
    logging.info('Space reduction:')
    logging.info('-'*72)
    for k in constraints:
        logging.info(k)
        c = len(constraints[k])
        e = len(exclusions[k])
        t = c + e
        r = float(c)/float(t)
        sr = r*r
        logging.info('\tPermutation length:\t{}'.format(t))
        logging.info('\tConstraint length:\t{}'.format(c))
        logging.info('\tExclusions length:\t{}'.format(e))
        logging.info('\tLinear ratio:\t\t{:.2f} ({}/{})'.format(r, c, t))
        logging.info('\tSquared ratio:\t\t{:.2f}'.format(sr))
        
    return (lab_to_id, id_to_lab, tag_to_span_fn, num_tags, constraints)
'''        