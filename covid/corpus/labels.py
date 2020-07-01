import re
import pandas as pd
import numpy as np
import re
from corpus.brat import HAS_ATTR

from constants import *

from utils.misc import list_to_nested_dict, list_to_dict
from utils.seq_prep import add_BIO_doc
from annotate.standoff_create import unmerge_entities, combine_seq, rename_entity


def update_event_counts(a, b):

    # Loop on events in b
    for k, cnt in b.items():
        
        # Add event to a
        if k not in a.keys():
            a[k] = 0

        # Increment Count
        a[k] += cnt
            
    return a
    

def event_counts_to_df(counts, events=None, entities=None):

    # Format counts as data frame
    count_list = []
    for (evt, ent), cnt in counts.items():
        count_list.append((evt, ent, cnt))

    # Convert to data frame
    cols = ['event', 'entity', 'count']
    df = pd.DataFrame(count_list, columns=cols) 
    df.sort_values(['event', 'entity'], inplace=True)

    # Convert to pivot table
    idx = ["entity"]
    vals = ["count"]
    cols = ["event"]
    pt = pd.pivot_table(df, index=idx, values=vals, columns=cols)    
    pt = pt.fillna(0)
    
    if entities is not None:
        pt = pt.reindex_axis(entities, axis=0)
    if events is not None:
        pt = pt.reindex_axis([("count", evt) for evt in events], axis=1)
    return pt

    

def counts_dict_to_df(ocur_counts, word_counts):

    # Convert to data frame
    X = []
    for event, entities in ocur_counts.items():
        for entity, _ in entities.items():
            oc = ocur_counts[event][entity]
            wc = word_counts[event][entity]
            X.append((event, entity, oc, wc))
    columns = ['event', 'entity', 'occurrence count', 'word count']
    df = pd.DataFrame(X, columns=columns) 
    
    return df    

def label_counts(events, outside=OUTSIDE):
    '''
    Get a label counts
    '''

    # Convert to nested dictionary if list
    if isinstance(events, list):
        events = list_to_dict(events)

    # Initialize counter repository    
    counts = {}

    # Loop on entities
    for event, labs in events.items():
               
        # Flatten word-level labels
        if isinstance(labs[0], list):
            labs = [tok for sent in labs for tok in sent]

        # Loop on individual labels
        for lb in labs:
            if (event, lb) not in counts.keys():
                counts[(event, lb)] = 0
            counts[(event, lb)] += 1

    return counts


def label_counts_nested(events, outside=OUTSIDE):
    '''
    Get a label counts
    '''

    # Convert to nested dictionary if list
    if isinstance(events, list):
        events = list_to_nested_dict(events)

    # Initialize counter repository    
    counts = {}
    
    # Loop on events
    for event, entities in events.items():
        
        # Loop on entities
        for entity, labs in entities.items():
                   
            # Flatten word-level labels
            if isinstance(labs[0], list):
                labs = [tok for sent in labs for tok in sent]

            # Loop on individual labels
            for lb in labs:
                if (event, (entity, lb)) not in counts.keys():
                    counts[(event, (entity, lb))] = 0
                counts[(event, (entity, lb))] += 1

    return counts



def process_predictions(Y):  
    '''
    Process note
    '''

    '''
    Output in BRAT format
    '''

    Y_ = []
    
    for y in Y:


        # Remove "helper" events
        if y is not None:
            y = {evt:ents for evt, ents in y.items() if evt not in [SUBSTANCE]}

        # Unmerge entities, e.g. Amount, Frequency, etc. merged Alcohol
        y = unmerge_entities(y, SEQ_TAGS, OUTSIDE)
       
        # Combine sentence-level and word-level status labels
        y = combine_seq(labels = y, 
                         sent_entity = STATUS, 
                         seq_entity = STATUS_SEQ, 
                         sent_neg_label = OUTSIDE, 
                         seq_pos_label = 1)


        y = rename_entity(y, INDICATOR_SEQ, new_=None)
        

        if y is not None:
            y = {evt:{ent:labs for ent, labs in ents.items() if ent not in [INDICATOR, INDICATOR_SEQ, STATUS_SEQ]} for evt, ents in y.items()}

        Y_.append(y)

    return Y_








def merge_labels(y, hierarchy=None, outside=OUTSIDE):
    '''
    Merge labels for single token
    
    args: 
        y = list of labels for given token
        hierarchy = negative label (outside label) should be the first 
        index and the highest priority label should be the last index
    '''

    # List of unique labels
    unique = list(set(y))
    
    # Empty labels
    if len(unique) == 0:
        return outside

    elif len(unique) == 1:
        return unique[0]
    
    # No hierarchy provided
    elif hierarchy is None:

        if (len(unique) == 2) and (outside in unique):
            unique.remove(outside)
            return unique[0]
        else:
            raise ValueError("Cannot merge: {}".format(unique))

    # Hierarchy provided
    else:
        
        # Find highest priority label
        for h in hierarchy[::-1]:
            if h in unique:
                return h

    raise ValueError("Could not merge labels")



def merge_labels_(y, hierarchy):
    '''
    Merge labels for single token
    
    args: 
        y = list of labels for given token
        hierarchy = the highest priority label should be the last index
    '''

    # Get unique labels
    unique = set(y)
    
    # Find highest priority label
    for h in hierarchy[::-1]:
        if h in unique:
            return h

    raise ValueError("Could not merge labels")



def doc_level_labels(labels, events, entity, hierarchy):
    '''
    Get document-level labels
    '''
    
    # Initialize document labels as None
    doc_labels = {event:hierarchy[-1] for event in events}
    
    # No labels
    if labels is None:
        return doc_labels
        
    # Loop on events
    for event in events:
        
        # Initialize sentence labels to lowest priority label in 
        # hierarchy
        sent_labs = [hierarchy[-1]]
        
        # Get sentence label
        for sent in labels:

            if sent is not None:

                sent_labs.append(sent[event][entity])
    
        # Merge sentence-level labels to document-level label    
    
        # Merge sentence-level labels to document-level label
        doc_labels[event] = merge_labels(sent_labs, OUTSIDE, hierarchy)
    
    return doc_labels            


    
def doc_keyword_ind(labels, tokens, event, entity, label, pattern):
        
    # No labels        
    if labels is None:
        return 0

    # Check length
    assert len(labels) == len(tokens), "Length mismatch"

    # Loop on sentences
    found = []
    for sent_labs, sent_tok in zip(labels, tokens):
        
        if sent_labs is not None:
        
            # Get specified labels
            sent_labs_ = sent_labs[event][entity]

            # Check len()
            assert len(sent_labs_) == len(sent_tok), 'Length mismatch'
            
            # Loop on tokens and sentence
            for tok_lab, tok_tok in zip(sent_labs_, sent_tok):
                if tok_lab == label:
                    found.append(tok_tok)
    
    # Convert to string
    found = " ".join(found)
    
    # Search for pattern
    match = re.search(pattern, found, re.I)
    
    # Convert match to binary indicator
    indicator = int(bool(match))

    return indicator

def binarize_seq(labels, neg_label, reduce_dim):

    # Get type
    is_list = isinstance(labels, list)

    # Convert to numpy array
    labels = np.array(labels)

    # Binarize
    labels = labels != neg_label

    # Cast as integer 
    labels = labels.astype(int)
     
    # Convert from sequence of sequence to sequence
    if reduce_dim:
        labels = int(np.any(labels, axis=0))

    # Convert back to list
    if is_list:
        labels = list(labels)
            
    return labels



def binarize_doc(labels, neg_label, reduce_dim):

    # Get binary document labels
    binary_labels = []
    for sent in labels:

        # Append document
        binary_labels.append(binarize_seq(sent, neg_label, reduce_dim))
            
    return binary_labels    
    
    


def get_span_label(event_type, span_type, span_label, label_type, 
            merge_substance=False):
    
    '''
    Get span
    '''
    # New label
    if label_type == EVENT_TYPE:
        y = event_type
    elif label_type == SPAN_TYPE:
        y = span_type
    elif label_type == EVENT_SPAN_TYPE:
        y = '{}-{}'.format(event_type, span_type)
    elif label_type == EVENT_LABEL:
        y = '{}-{}'.format(event_type, span_label)
    elif label_type == LABEL:
        y = span_label
    else: 
        raise ValueError( \
             "Invalid label_type:{}".format(label_type))

    if merge_substance:
        y = re.sub("|".join(SUBSTANCES), SUBSTANCE, y)

    return y
    