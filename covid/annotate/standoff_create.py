import sys
import os
import zipfile
import glob
from collections import namedtuple
import re
import numpy as np
from six import iteritems
import json
import joblib
import json

import warnings
import difflib
import copy
import warnings
import shutil
import tarfile
from difflib import SequenceMatcher

from collections import Counter
import pandas as pd
from utils.tokenization_nltk import tokenize_doc
from utils.misc import dict_to_list
from utils.seq_prep import preprocess_labels_doc, preprocess_tokens_doc
from utils.seq_prep import OUTSIDE, NEG_LABEL
from collections import namedtuple
from more_itertools import unique_everseen

from annotate.standoff_load import TEXT_FILE_EXT, ANN_FILE_EXT, ENTITY_MAP_CREATE
from corpus.alignment import align

Index = namedtuple('Index', ['start', 'stop'])
Span = namedtuple('Span', ['event', 'entity', 'start', 
                                           'stop', 'text', 'attribute'])

from constants import ANNOTATION_CONF, VISUAL_CONF



def to_standoff_corpus(dest, fns, text, tokens, labels, has_attribute):
    '''
    Create BRAT txt and ann files for corpus
    
    args:
        dest = destination folder
        
        fns = filename associated with each sentence in document 
            (filename without path or ext)
        text = list of documents, 
            where each document is represented as strings
        tokens = corpus represented as nested lists of strings
            i.e., each document is represented as 1 or more lists of str
        labeled = corpus represented as nested list of dictionaries
        has_attribute = list of entities that include attributes
    '''
    

    
    # Check document count again  
    assert len(fns) == len(text), \
                               "text and unique filename count mismatch"
      
    # Loop on documents in corpus
    for fn, txt, tok, lab in zip(fns, text, tokens, labels):
        if True or (fn == 175):
            to_standoff_doc(dest, fn, txt, tok, lab, has_attribute)
            

    return True 

def group_by_doc(fns, X):
    '''
    args:
        fns = list of filenames
        X = list of same size as fns
    
    '''
    
    # Check size
    assert len(fns) == len(X), \
    '''length mismatch: fns cnt={}, X cnt={}'''.format(len(fns),len(X))
    
    # Get list of unique filenames, preserving order
    #fns_unique = list(unique_everseen(fns))
    
    indices = []
    start = 0
    #cnt = len(fns)
    for current, f in enumerate(fns):       
    
        # If fn changed append indices
        if (f != fns[start]):
            indices.append((start, current))
            start = current

    # Include last index
    indices.append((start, len(fns)))
       
    # Group by filename
    fns_unique = []
    docs = []
    for start, stop in indices:

        # Append elements associated with current document
        fns_unique.append(fns[start])
        docs.append(X[start:stop])

    # Count totals
    l1 = len(X)
    l2 = sum([len(x) for x in docs])
    
    assert l1 == l2, "length error: {}, {}".format(l1, l2)
    assert len(fns_unique) == len(indices), (len(fns_unique), len(indices))        
    
    return (fns_unique, docs)

def to_standoff_doc(dest, fn, text, tokens, labels, has_attribute):
    '''
    Create BRAT txt and ann files for single document
    
    args:
        dest = destination folder
        fn = filename for document, without path or extension

        text, tokens, and labels are lists of document-level values
        text = documents represented as strings
        tokens = documents represented as nested lists of strings
        labeled = documents represented as nested list of dictionaries
        has_attribute = list of entities that include attributes
    '''

    indices = align(text, tokens)

    # Check number of sequences
    assert len(tokens) == len(labels), 'sequence count mismatch'

    # Loop on sentences    
    spans = []
    for i, (toks, labs, indxs) in \
                                enumerate(zip(tokens, labels, indices)):
        
        if not (labs is None):
            
            # Check number of tokens/labels and sentence 
            for evt, ents in labs.items():
                for ent, lb in ents.items():
                    assert len(toks)==len(lb), \
                    '''sequence length mismatch: 
                       tokens count = {}, tokens = {}
                       labeel count = {}, labels = {}'''.format( \
                            len(toks), toks, len(lb), lb)
            
            # Append spans for current sentence
            spans.append(get_spans_sent(text, labs, indxs, has_attribute))
    '''
    if 'ETOH, lives in [**Location 686**]' in text:
        for i, sent in enumerate(spans):
            print(i)
            for evt, spn in sent.items():
                print(evt)
                for s in spn:
                    print(s)
                
                
        for i, sent in enumerate(labels):
            print(i)
            for evt, ents in sent.items():
                
                print(evt)
                for ent, lbs in ents.items():
                    print(ent, lbs)
    '''


    # Create annotation string
    ann = create_ann_str(spans)
        
    write_brat(dest, fn, text, ann)
    
    #print(dest)

    # Create compressed tar file
    #fn = os.path.join(os.path.dirname(dest), \
    #                              os.path.basename(dest)+'.tar.gz')

    return True 

def write_brat(path, fn, text, ann):
    
    # Write files
    fn_ann = os.path.join(path, '{}.{}'.format(fn, ANN_FILE_EXT))
    with open(fn_ann,'w') as f:
        f.write(ann)

    fn_txt = os.path.join(path, '{}.{}'.format(fn, TEXT_FILE_EXT))
    with open(fn_txt,'w') as f:
        f.write(text)   

    return True 

def flatten(X):
    return [i for x in X for i in x]


 
    
def get_spans_sent(text, labels, indices, has_attribute):
    '''
    Determine word indices of spans in sentence
    
    args:
    
    returns:
        dictionary of list of spans as tuple, e.g.
        {'Drug': 
            [Span(entity='Amount', start=26, stop=33, text='abuse .', attribute=None)], 
         'Alcohol': 
            [Span(entity='Amount', start=26, stop=33, text='abuse .', attribute=None)], 
         'Tobacco': []}

        
    '''
       
    span_dict = {}
    for event, entities in labels.items():
        span_dict[event] = []
        for entity, labs in entities.items():

            # Get spans with word-level start and stop indices  
            has_att = entity in has_attribute
            spans = get_spans_label(text, labs, indices, \
                                               has_att, event, entity)

            # Aggregate
            span_dict[event].extend(spans)

    # Check if non-head entities
    has_entities = []
    for event in span_dict:
        for span in span_dict[event]:

            # Non-head entity found
            if span.entity != event:
                has_entities.append(event)
                break
    
    span_dict_filt = {}
    for evt in has_entities:
        span_dict_filt[evt] = span_dict[evt]
      
    
    return span_dict_filt



def get_spans_label(text, labels, indices, has_attribute, \
                                        event, entity, \
                                        outside = OUTSIDE,
                                        neg_label = NEG_LABEL):
    '''
    Get indices of non-negative span labels for single sequence
    
    args:
        text = doc text as string
        labels = list of labels
        has_attribute = Boolean indicating whether attribute included
        entity = entity associated with event
        outside = outside label (e.g. 'O')
        
    returns:
        spans = list of spans, where index is the position in the sentence 
                (not character index)
    '''
    
    # Shift sequence to assess adjacent labels
    labels_prev = [outside] + labels[0:-1]
    labels_next = labels[1:] + [outside]

    # Get new labels
    spans = []
    for i, (prev, curr, nxt) in \
                       enumerate(zip(labels_prev, labels, labels_next)):
        
        # Current label positive?
        is_positive = curr not in [outside, neg_label]
        
        # Current label not equal to previous (rising edge)
        is_rising = curr != prev
        
        # Current label not equal to next (falling edge)
        is_falling = curr != nxt

        # Positive rising edge (non-negative label)        
        if is_positive and is_rising:
            
            # Current, positive label
            label = curr
            
            # Index in sequence of span start (token index)
            start_tok = i 
        
        # Positive falling edge, so construct span
        if is_positive and is_falling:
            
            # Index in sequence of span stop
            stop_tok = i
                        
            # Span character indices
            
            assert (start_tok < len(indices)) \
                   and (stop_tok < len(indices)), '''
            Text = {}
            Labels = {}
            Indices = {}
            start_tok = {}
            stop_tok = {}
            '''.format(text, labels, indices, start_tok, stop_tok)
                        
            start_char = indices[start_tok][0]
            stop_char = indices[stop_tok][1]

            # Span text
            txt = text[start_char:stop_char]
            
            if not ('\n' in txt):
                # Build span
                spans.append(Span( \
                        event = event,
                        entity = entity, 
                        start = start_char, 
                        stop = stop_char,
                        text = txt,
                        attribute = label if has_attribute else None))  
            else:
                # Split spans across linebreaks 
                txt_split = txt.split('\n') 
                #print("get_spans_label", 'txt_split')
                #print('txt:', txt)
                #print('txt_split:', txt_split)
                indices_new = align(txt, [txt_split])
                
                indices_new = indices_new[0]
                
                for (start, stop), t in zip(indices_new, txt_split):
                    
                    # Build span
                    spans.append(Span( \
                          event = event,
                          entity = entity, 
                          start = start + start_char, 
                          stop = stop + start_char,
                          text = t,
                          attribute = label if has_attribute else None))  

                #print(spans[-1])

    return spans

    
def textbound_str(id_, span):
    return 'T{id_}\t{entity} {start} {stop}\t{text}'.format( \
        id_ = id_, 
        entity = span.entity, 
        start = span.start, 
        stop = span.stop, 
        text = span.text)

def event_str(id_, textbounds, event):
    '''
    Create event string
    
    args:
        id_ = ID for event
        textbounds = Applicable text bounds
    '''
    
    # Start event string
    out = 'E{}\t'.format(id_)
    
    # Get text bound associated with event
    event_tb = [(entity, id_) for (entity, id_) in textbounds \
                                                  if entity == event]
    # If multiple events    
    if len(event_tb) > 1:
        msg = '''Multiple events found:
            id_ = {}
            textbounds = {}
            event = {}'''.format(id_, textbounds, event)
        warnings.warn(msg)

    
    if len(event_tb) == 0:
        tb = (event, 1)
        msg = 'No event found. Defaulting to 1st textbound:\t{}'.format(tb)
        warnings.warn(msg)
        textbounds.insert(0, tb)
    else:
        event_tb = event_tb[0]

        # Force event text bound to be first
        textbounds.remove(event_tb)
        textbounds.insert(0, event_tb)

        
    # Append text bounds to event string
    for entity, id_ in textbounds:
        out += '{}:T{} '.format(ENTITY_MAP_CREATE.get(entity, entity), id_)
        
    return out

def attr_str(id_, tb, value):
    return 'A{id_}\tValue T{tb} {value}'.format( \
        id_ = id_, 
        tb = tb, 
        value = value)
    

    
def create_ann_str(spans):

    """
    Parse textbound annotations in input, returning a list of
    Textbound.

    args:
        spans = list of dictionary of spans

    ex output. 
        T1	Status 21 29	does not
        T1	Status 27 30	non
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
        T7	Type 78 90	recreational
        T8	Drug 91 99	drug use
        
        T8\tDrug 91 99\tdrug use


        E4      Tobacco:T12 State:T11
        E5      Alcohol:T15 State:T14
        E6      SexualHistory:T18 Type:T19 Time:T17 State:T16
        E1      Family:T1 Amount:T2 Type:T3
        E2      MaritalStatus:T4 State:T6 Type:T5
        E1      Tobacco:T2 State:T1
        E1      Tobacco:T2 State:T1
        E2      Alcohol:T5 State:T4
        
        E2\tAlcohol:T5 State:T4
        
        A2      Value T8 current
        A3      Value T9 none
        A4      Value T13 current
        
        A4\tValue T13 current

    """

    # Initialize indices for text bounds, events, and attributes
    iT = 1
    iE = 1
    iA = 1
    
    # Initialize list of invitation strings
    output = []
    
    # Loop on sentences
    for sent in spans:
        
        # Loop on events and spans    
        for event, S in sent.items():
            
        
            tbs = []
            attrs = []
            
            # Loop on spans for current event
            for s in S:

                output.append(textbound_str(iT, s))
                tbs.append((s.entity, iT))
            
                if s.attribute is not None:
                    output.append(attr_str(iA, iT, s.attribute))

                    iA += 1
                iT += 1
            
            if len(tbs) > 0:
                output.append(event_str(iE, tbs, event))
                
                iE += 1
           
    output = "\n".join(output)
    
    return output



def unmerge_entities(labels, merged_entity, outside):   
#def unmerge_entities(labels, events, entities, merged_entity, outside):    
    '''
    Unmerge entities for a single sentence
    
    args: 
        labels = list of dict of labels for each sentence
        events = events to modify (e.g. [Alcohol, Drug...])
        entities = entities that were merged (e.g. [Amount, Frequency...])
        merged_entity = name of merged entity (e.g. Quantity)
        outside = outside label (e.g. 'O')
    '''

        
    # No labels, perpetuate value of None
    if labels is None:
        return None
    
    # Labels found
    else:

        # Initialize new sentence labels
        new_labels = copy.deepcopy(labels)

        # Loop on events (e.g. Alcohol, Drug, Tobacco)
        for evt in labels:

            # Get all entities
            entities = set(labels[evt][merged_entity])
            if outside in entities:
                entities.remove(outside)

            # Loop on unmerged entities (e.g. Amount, Frequency, etc.)
            for ent in entities:
                               
                # Loop on entity labels in sentence
                L = []
                for entity_label in labels[evt][merged_entity]:
                    
                    if entity_label == ent:
                        L.append(ent)
                    else:
                        L.append(outside)
                
                new_labels[evt][ent] = L
            
            # Remove merged label
            del new_labels[evt][merged_entity]
                   
        return new_labels
    
    
'''
def unmerge_events(labels, events, entity, merged_event, outside):
    
    Unmerge events
    

    if labels is None:
        return None
        
    # Labels found
    else:

        # Initialize new sentence labels
        new_labels = copy.deepcopy(labels)

        # Loop on events (e.g. Alcohol, Drug, Tobacco)
        for event in events:

            # Loop on entity labels in sentence
            L = []
            for entity_label in labels[(merged_event, entity)]:
                
                if re.search(event, entity_label):
                    L.append(entity)
                else:
                    L.append(outside)
            
            new_labels[(event, entity)] = L
            
        # Remove merged label
        del new_labels[(merged_event, entity)]
                
    
        return new_labels
'''

def combine_seq(labels, sent_entity, seq_entity, sent_neg_label, \
                seq_pos_label):
    '''
    Combine sentence and a word-level labels, e.g. Status
    
    args:
        labels = list of sentence labels
        events = list of events of interest (e.g. [Alcohol, Drug...])
        sent_entity = sentence level entity (e.g. 'Status')
        seq_entity = word-level entity (e.g. 'StatusSeq')
        neg_label = negative label label (e.g. 'O' or 0)
    '''
    
    
    if labels is None:
        return None
        
    # Labels found
    else:

        # Initialize new sentence labels
        new_labels = copy.deepcopy(labels)
        
        # Loop on events (e.g. Alcohol, Drug, Tobacco)
        for event, ents in labels.items():

            seq = ents[seq_entity]
            sent = ents[sent_entity]

            # Initialize sequence to negative label
            L = [sent_neg_label]*len(seq)
            
            # If non-negative label at sentence level, update 
            #   sequence labels
            if sent != sent_neg_label:
                for i in range(len(L)):
                    if seq[i] == seq_pos_label:
                        L[i] = sent

            new_labels[event][sent_entity] = L
        del new_labels[event][seq_entity]
        return new_labels

        '''
            # Combine all labels for all entities for current event
            # into one list
            all_labs = []
            for (evt, ent), labs in labels.items():
                if (evt == event) and (ent != seq_entity):
                    if isinstance(labs, list):
                        all_labs.extend(labs)
                    else:
                        all_labs.append(labs)
            
            # Determine if any positive labels present
            unique_labs = set(all_labs)
            positive_labs = unique_labs - set(neg_labels)
            any_positive = len(positive_labs) > 0
            

            # Loop on entity labels in sentence
            L = []
            for entity_label in labels[(event, seq_entity)]:
                
                # Current position has negative label
                if entity_label in neg_labels:
                    L.append(neg_label)
                    #if check_pos:
                    #    print(1)        
                # Current position has positive label AND
                # check_pos == True AND
                # at least one entity has at least on positive label
                elif check_pos and any_positive:
                    L.append(pos_label)
                    #if check_pos:
                    #    print(2)
                elif check_pos and not any_positive:
                    L.append(neg_label)
                    #if check_pos:
                    #    print(3)
                
                # Current position is positive
                else:
                    L.append(labels[(event, sent_entity)])
                    #if check_pos:
                    #    print(4)
            new_labels[(event, sent_entity)] = L
            
            # Remove merged label
            del new_labels[(event, seq_entity)]
        return new_labels
        '''
        
        


    
    
def remove_labels(labels, events, entities):
    '''
    Remove event-entity combinations
    '''

    # Create new document
    if labels is None:
        return None
        
    # Labels found
    else:

        # Initialize new sentence labels
        new_labels = {}

        # Loop on sentence labels
        for (event, entity), labs in labels.items():

            # Include event-entity labels, \
            # if both are not in remove list
            if (event not in events) and (entity not in entities):
            
                # Remove merged label
                new_labels[(event, entity)] = \
                                    copy.deepcopy(labels[(event, entity)])
                
    
        return new_labels


 


def rename_entity(labels, orig, new_=None):
    '''
    Rename entity
    '''

    if labels is None:
        return None
        
    # Labels found
    else:

        # Initialize new sentence labels
        new_labels = copy.deepcopy(labels)

        # Create new entity name
        for evt in labels:
            if new_ is None:
                ent_new = evt
            else:
                ent_new = new_
           
            new_labels[evt][ent_new] = \
                           copy.deepcopy(new_labels[evt][orig])
            
        # Remove original entity
        del new_labels[evt][orig]
                    
        return new_labels



#
#def get_label_subset(labels, bools, keep=True):

#    '''
#    Get subset of labels, based on Boolean
#    '''
#    label_subset = {}
#    for (entity, type_), labs in labels.items():
#        if bools[entity] == keep:
#            label_subset[(entity, type_)] = labs
#    
#    return label_subset  
#
#
#def get_seq_labels(labels, is_seq):
#    '''
#    Get sequence labels only
#    '''
#    return get_label_subset(labels, is_seq, keep=True)  
#    
#def get_sent_labels(labels, is_seq):
#    '''
#    Get sequence labels only
#    '''
#    return get_label_subset(labels, is_seq, keep=False)  
