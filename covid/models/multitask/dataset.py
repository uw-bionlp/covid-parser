
from collections import OrderedDict, Counter

import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import sys
import os
import errno
from datetime import datetime
from tqdm import tqdm
import numpy as np
import logging
from tqdm import tqdm
import joblib
import math


from utils.misc import nested_dict_to_list, list_to_nested_dict, dict_to_list
from utils.seq_prep import preprocess_tokens_doc
from models.crf import add_bio_labels, BIO_to_span
from models.xfmrs import tokens2wordpiece, get_embeddings, embed_len_check, get_embeddings_preloaded
from models.utils import pad_embedding_seq, mem_size, pad_sequences, create_mask,  seq_label,  map_1D, map_2D, map_dict_builder
from corpus.event import Event, Span, events2sent_labs, events2seq_tags

from constants import *



def get_label_map(label_def):


    # Initialize output
    to_id = OrderedDict()
    to_lab = OrderedDict()
    num_tags = OrderedDict()
    
    # Loop on label categories
    for name, lab_def in label_def.items():
        
        to_id[name] = OrderedDict()
        to_lab[name] = OrderedDict()
        num_tags[name] = OrderedDict()
        
        # Loop on argument types in label map
        for evt_typ, map_ in lab_def[LAB_MAP].items():
            
            # Account for BIO prefixes
            if lab_def[LAB_TYPE] == SEQ:
                map_ = [(OUTSIDE, map_[0])] + \
                       [(p, m) for m in map_[1:] for p in [BEGIN, INSIDE]]
            
            # Generate map
            to_id[name][evt_typ], to_lab[name][evt_typ] = \
                                                  map_dict_builder(map_)
            num_tags[name][evt_typ] = len(map_)

    logging.info("Label to ID mapping functions:")
    for name, arguments in to_id.items():
        logging.info('\t{}'.format(name))
        for evt_typ, map_ in arguments.items():
            logging.info('\t\t{}'.format(evt_typ))
            for k, v in map_.items():
                logging.info('\t\t\t{} --> {}'.format(k, v))
            
    logging.info("ID to Label mapping functions:")
    for name, arguments in to_lab.items():
        logging.info('\t{}'.format(name))
        for evt_typ, map_ in arguments.items():
            logging.info('\t\t{}'.format(evt_typ))
            for k, v in map_.items():
                logging.info('\t\t\t{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Number of tags:')
    for name, lab_def in num_tags.items():
        logging.info('\t{}'.format(name))
        for name, cnt in lab_def.items():
            logging.info('\t\t{} = {}'.format(name, cnt))


    return (to_id, to_lab, num_tags)



def events2multitask_labs(events, label_def, max_len, pad_start, \
                               label_to_id=None, include_prefix=True):
    '''
    Get multitask labels from events
    
    Parameters
    ----------
    events: document labels as sequence of sentences as sequence of Event
            e.g. [[Event(), ... Event()], 
                  [Event(), ... Event()],
                  [Event(), ... Event()]]

    label_map: nested event type-argument type combinations as dict of list 
            e.g. {'Trigger: 
                    {'Alcohol': ['Outside', 'Alcohol']},...
                 {'Status: 
                    {'Alcohol': ['Outside', 'none',...]},...    
                 {'Entity': 
                    {'Alcohol': ['Amount', 'Frequency']},... 
                    }
    
    max_len: maximum sequence length as int

    pad_start: Boolean indicating whether start of sequence is padded 
                with start of sequence token

    
    Returns
    -------
    trig: 
    status: 
    entity: 
    prefixes: 
                   
    '''    
    
    # Sentence count
    n = len(events)

    # Initialize output
    y = [OrderedDict() for _ in range(n)]
    
    # Iterate over label types
    for name, lab_def in label_def.items():

        lab2id = None if label_to_id is None else label_to_id[name]
        
        # Sentence-level label
        if lab_def[LAB_TYPE] == SENT:
            y_tmp = events2sent_labs( \
                        events = events, 
                        label_map = lab_def[LAB_MAP], 
                        arg_type = lab_def[ARGUMENT],
                        label_to_id = lab2id)        
                
        # Token-level labels
        elif lab_def[LAB_TYPE] == SEQ:
            y_tmp = events2seq_tags( \
                        events = events, 
                        label_map = lab_def[LAB_MAP], 
                        max_len = max_len, 
                        pad_start = pad_start,
                        include_prefix = include_prefix,
                        label_to_id = lab2id)

        else:
            raise ValueError("Invalid label type:\t{}".format(lab_def[LAB_TYPE]))


        assert len(y_tmp) == len(y)
        for i, y_ in enumerate(y_tmp):
            y[i][name] = y_
            
    return y
  


def preprocess_X_xfmr(X, \
                xfmr_type = None, 
                xfmr_dir = None,
                device = None,
                max_len = 30, 
                num_workers = 6,
                get_last = True,
                batch_size = 50,
                embedding_model = None,
                tokenizer_model = None
                ):

    '''
    Preprocess tokenized input text
    
    Parameters
    ----------
    X: list of tokenized sentences, 
            e.g.[['Patient', 'denies', 'tobacco', '.'], [...]]
    
    '''    
    
    # Flatten documents into sequence of sentences
    X = [sent for doc in X for sent in doc]
    
    # Get word pieces
    wp_toks, wp_ids, tok_idx = tokens2wordpiece( \
                            tokens = X, 
                            xfmr_type = xfmr_type, 
                            xfmr_dir = xfmr_dir, 
                            get_last = get_last,
                            tokenizer = tokenizer_model)

    # Get sequence length, with start and and padding
    seq_lengths = [len(x) for x in tok_idx]
    
    # X as embedding
    embed = get_embeddings_preloaded(embedding_model,
                            word_piece_ids = wp_ids, 
                            tok_idx = tok_idx, 
                            #xfmr_type = xfmr_type, 
                            #xfmr_dir = xfmr_dir, 
                            num_workers = num_workers,
                            batch_size = batch_size,
                            device = device)
    # Check lengths
    embed_len_check(X, embed)

    # Pad sequences of embedding
    embed = [pad_embedding_seq(x, max_len) for x in embed]    
    
    # Convert embeddings to tensor
    embed = torch.tensor(embed)
              
    return (X, embed, seq_lengths)    


def preprocess_X_w2v(X, \
                to_lower = True, 
                embed_map = None,                
                embed_matrix = None, 
                freeze = True, 
                max_len = 30, 
                pad_start = True, 
                pad_end = True, 
                num_workers = 6,
                unk = UNK, 
                start_token = START_TOKEN, 
                end_token = END_TOKEN):


    '''
    Preprocess tokenized input text
    
    Parameters
    ----------
    X: list of tokenized sentences, 
            e.g.[['Patient', 'denies', 'tobacco', '.'], [...]]
    
    '''    

        
    logging.info('='*72)
    logging.info('Preprocessing X, w2v')
    logging.info('='*72)
    
    # Flatten documents into sequence of sentences
    X = [sent for doc in X for sent in doc]
    logging.info('\tDocument count:\t{}'.format(len(X)))
    logging.info('\tFlattened documents to sequence of sentences')
    logging.info('\tSentence count:\t{}'.format(len(X)))
    
    # Convert to lowercase
    if to_lower:       
        X = [[tok.lower() for tok in sent] for sent in X]
        logging.info('\tConverting to lowercase. Ex.')
        for x in X[0:5]:
            logging.info('\t\t{}'.format(x))
    else:
        logging.warn('\tNOT converting to lowercase')
    
    # Include start and end of sequence padding tokens 
    # (still variable length)
    tokens_proc = preprocess_tokens_doc(X, \
                    pad_start = pad_start,
                    pad_end = pad_end,
                    start_token = start_token,
                    end_token = end_token,
                    )

    # Get sequence length, with start and and padding
    seq_lengths = [len(x) for x in tokens_proc]

    # Map to token IDs
    ids = map_2D(tokens_proc, embed_map)

    # Pad so fixed length sequence
    ids = pad_sequences(ids, max_len)
    ids = torch.LongTensor(ids).to(embed_matrix.weight.device)
    
    embed = embed_matrix(ids)
    
    logging.info('')
    return (X, embed, seq_lengths)    



def preprocess_y(events_by_doc, label_def, max_len, pad_start, label_to_id):
    '''
    Convert events to multitask label IDs
    '''
    
    # Bail if no input    
    if events_by_doc is None:
        return events_by_doc

    # Get summary of input events
    events_flat = [event for doc in events_by_doc for sent in doc for event in sent]
    event_counts = len(events_flat)
    span_counts = sum([len(evt.arguments) for evt in events_flat])
    types_ = [(evt.type_, arg.type_) for evt in events_flat for arg in evt.arguments]
    types_ = list_to_counts(types_)
    
    logging.info('Event import summary: ')
    logging.info("Event counts:\t{}".format(event_counts))
    logging.info("Span counts:\t{}".format(span_counts))
    logging.info("Event and argument types:\n{}".format(types_))


    # Convert events to spans
    y = []

    for doc_events in events_by_doc:
        y_doc = events2multitask_labs( \
                                events = doc_events, 
                                label_def = label_def, 
                                max_len = max_len, 
                                pad_start = pad_start,
                                label_to_id = label_to_id,
                                include_prefix = True)    
        y.extend(y_doc)


    # Get summary of encoded events
    counts = []
    for y_ in y:
        for name, lab_def in y_.items():
            for evt_typ, labs in lab_def.items():
                
                label_type = label_def[name][LAB_TYPE]
                if label_type == SENT:
                    counts.append((name, evt_typ, labs))
                elif label_type == SEQ:
                    counts.extend([(name, evt_typ, lb) for lb in labs])
                else:
                    raise ValueError("invalid label type:\t{}".format(label_type))
    counts = list_to_counts(counts)                    
    logging.info("Encoded event summary:\n{}".format(counts))


    for i, y_ in enumerate(y):
        for name, evt_labs in y_.items():
            for evt_typ, labs in evt_labs.items():
                if isinstance(labs, list):
                    y[i][name][evt_typ] = torch.LongTensor(labs)

    return y





def sent_lab_span(labels, tokens, pad_start, id_to_label, type_, sent_idx):
    

    # Sequence length
    n = len(tokens)
    
    # Iterate over labels in sequence
    for i, lab in enumerate(labels):
        
        # Find first non-negative label
        if lab > 0:
            
            # Position of trigger start
            start = i - int(pad_start)
            start = min(max(start, 0), n-1)
            end = start + 1
            
            # Create span
            span = Span( \
                            type_ = type_, 
                            sent_idx = sent_idx, 
                            tok_idxs = (start, end), 
                            tokens = tokens[start:end], 
                            label = id_to_label[lab])         
            return span

    # Return None, if nothing detected
    return None

def seq_tag_spans(labels, tokens, pad_start, id_to_label, sent_idx):


    # Get tags and span indices from BIO
    tag_start_end = BIO_to_span(labels, id_to_label)
    
    # Loop on spans
    spans = []
    for type_, start, end in tag_start_end:
        
        # Decrement token indices, if start padded
        start -= int(pad_start)
        start = max(start, 0)
        end -= int(pad_start)
        end = max(end, 0)
        
        spn = Span( \
                        type_ = type_, 
                        sent_idx = sent_idx, 
                        tok_idxs = (start, end), 
                        tokens =  tokens[start:end], 
                        label = None)
        spans.append(spn)
            
    return spans
    

def decode_(y, tokens, id_to_label, pad_start, label_def):
    '''
    Postprocess predictions
    '''

   
    # All possible event types
    evt_types = list(set([evt_typ for name, evt_labs in y[0].items() \
                                             for evt_typ in evt_labs]))
    # Loop on sentences
    assert len(y) == len(tokens)
    events_doc = []
    for i, (y_, toks) in enumerate(zip(y, tokens)):

        # Spans by event
        evt_spans = OrderedDict([(evt_typ, []) for evt_typ in evt_types])
        
        # Loop on label types (e.g. Trigger, Status, etc.)
        for name, evt_labs in y_.items():
            
            # Label type (e.g. 'sentence' or 'sequence')
            lab_typ = label_def[name][LAB_TYPE]
            
            # Loop on event types for label type (e.g. Alcohol, Drug, etc.)
            for evt_typ, labs in evt_labs.items():
            
                id2lab = id_to_label[name][evt_typ]    
            
                if lab_typ == SENT:
                    span = sent_lab_span( \
                                    labels = labs, 
                                    tokens = toks,
                                    pad_start = pad_start, 
                                    id_to_label = id2lab,
                                    type_ = name,
                                    sent_idx = i,
                                    )

                    if span is not None:
                        evt_spans[evt_typ].append(span)
                
                elif lab_typ == SEQ:
                    spans = seq_tag_spans( \
                                    labels = labs, 
                                    tokens = toks,
                                    pad_start = pad_start, 
                                    id_to_label = id2lab,
                                    sent_idx = i,
                                    )
                    evt_spans[evt_typ].extend(spans)
                else:
                    raise ValueError("invalid label type:\t{}".format(lab_typ))
            
        # Convert spans into events
        events_sent = []
        for evt_typ, spans in evt_spans.items():            

            # Build event
            if len(spans) > 0:
                evt = Event( \
                        type_ = evt_typ,
                        arguments = spans)
                events_sent.append(evt)
        
        events_doc.append(events_sent)                    
    
    return events_doc

def list_to_counts(X):

    counts = Counter(X)
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df = df.rename(columns={'index':'event', 0:'count'})
    return df

class MultitaskDataset(Dataset):
    """   
    """
    def __init__(self, X, y, label_def, \
        word_embed_to_lower = True,
        word_embed_map = None,
        word_embed_matrix = None,
        word_embed_freeze = True ,
        xfmr_type = None,
        xfmr_dir = None,
        xfmr_device = None,
        use_xfmr = False,
        max_len = 30,
        pad_start = True, 
        pad_end = True,
        num_workers = 6,        
        

        
    
        ):
        super(MultitaskDataset, self).__init__()

        '''
        Parameters
        ----------
        X: list of tokenized documents, 
                e.g. [doc [sent [tokens]]], [[['Patient', 'denies', 'tobacco', '.'], [...]]]
        y: list of documment events
                e.g. [doc [sent [events]]], [[[Event1, Event2, ...], [...]]]
        
        
        Returns
        -------

        '''
        
        self.label_def = label_def
        self.word_embed_map = word_embed_map
        self.xfmr_type = xfmr_type
        self.xfmr_dir = xfmr_dir
        self.use_xfmr = use_xfmr
        
        self.max_len = max_len + int(pad_start) + int(pad_end)
        self.pad_start = pad_start
        self.pad_end = pad_end
        self.num_workers = num_workers
        self.word_embed_to_lower = word_embed_to_lower
        self.word_embed_freeze = word_embed_freeze
        
        self.doc_count = len(X)
        self.sent_counts = [len(doc) for doc in X]       
        self.seq_count = sum(self.sent_counts)
        
                
        # Map to embeddings, and get sequence lengths
        if use_xfmr:
            self.tokens, self.X, self.seq_lengths = preprocess_X_xfmr( \
                    X = X, 
                    xfmr_type = xfmr_type,
                    xfmr_dir = xfmr_dir,
                    device = xfmr_device,
                    max_len = self.max_len, 
                    num_workers = self.num_workers,
                    get_last = True,
                    batch_size = 50,                   
                    )

        else:
            self.tokens, self.X, self.seq_lengths = preprocess_X_w2v( \
                    X = X, 
                    to_lower = word_embed_to_lower, 
                    embed_map = word_embed_map, 
                    embed_matrix = word_embed_matrix,
                    freeze = word_embed_freeze, 
                    max_len = self.max_len, 
                    pad_start = pad_start, 
                    pad_end = pad_end,
                    num_workers = self.num_workers)            
                    
        assert len(self.X) == self.seq_count
        
        
        self.mask = create_mask(self.seq_lengths, self.max_len)   


        # Get label maps
        self.label_to_id, self.id_to_label, self.num_tags = \
                                          get_label_map(self.label_def)

        if y is None:
            self.y = None
        else:
            # Process labeled input
            self.y = preprocess_y( \
                                        events_by_doc = y, 
                                        label_def = self.label_def, 
                                        max_len = self.max_len, 
                                        pad_start = self.pad_start,
                                        label_to_id = self.label_to_id)
             
        
    def __len__(self):
        return self.seq_count
        
    def __getitem__(self, index):
        
        # Current input and mask
        X = self.X[index]
        mask = self.mask[index]

        #Prediction (input only)
        if self.y is None:
            return (X, mask)
        
        # Supervised learning (input in labels)
        else:
            y = self.y[index]

            return (X, mask, y)



    def decode_(self, y):        


        pos_lab = []
        for y_ in y:
            for K, V in y_.items():
                for k, v in V.items():
                    if sum(v) > 0:
                        pos_lab.append((K, k))
        pos_lab = list_to_counts(pos_lab)
        logging.info('Label predictions for decoding. Positive counts:\n{}'.format(pos_lab))
        

        events = decode_( \
                            y = y, 
                            tokens = self.tokens,
                            id_to_label = self.id_to_label,
                            pad_start = self.pad_start,
                            label_def = self.label_def)
        
        event_counts1 = sum([len(sent) for sent in events])

        get_pos = lambda X: [k for x in X for k, v in x.items() for v_ in v if v_ > 0]


        
        logging.info("Multitask data set, decoding")
        logging.info("Event counts, by sent:\t{}".format(event_counts1))
        
        # Initialize output
        events_by_doc = [[[] for _ in range(c)] \
                                              for c in self.sent_counts]
        
        # Loop on documents
        i_sent = 0        
        for j_doc, cnt in enumerate(self.sent_counts):

            tmp = events[i_sent:i_sent + cnt]
            for j_sent, sent in enumerate(tmp):
                for evt in sent:
                    for span in evt.arguments:
                        span.sent_idx = j_sent
            events_by_doc[j_doc] = tmp
            
            i_sent += cnt


        event_counts2 = sum([len(sent) for doc in events_by_doc for sent in doc])
        assert event_counts1 == event_counts2, '{} vs {}'.format(event_counts1, event_counts2)

        return events_by_doc


