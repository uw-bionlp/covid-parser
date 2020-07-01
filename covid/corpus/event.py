import logging
from collections import OrderedDict, Counter
import pandas as pd
import numpy as np
import torch
import copy
import os

#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

from utils.seq_prep import strip_BIO_tok, is_begin, is_inside, is_outside
from utils.misc import to_flat_list
from utils.scoring import add_params_to_df
from utils.df_helper import filter_df
from models.utils import get_mapping_dicts, get_num_tags_dict, map_1D
from constants import *
from corpus.labels import merge_labels, merge_labels_
from utils.misc import list_to_dict, list_to_nested_dict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from constants import *

# From: https://www.ef.edu/english-resources/english-grammar/determiners/
IGNORE_TOKENS = ['the', 'a', 'an']

ANY_OVERLAP = 'any_overlap'
EXACT = 'exact'
PARTIAL = 'partial'
STATUS_MATCH = 'status_match'
CLOSEST = 'closest'
NO_DET = 'no_determiners'

INDEX_COLS = [EVAL_TYPE, EVENT_TYPE, ARGUMENT, LABEL]

GROUP = 'group'

RANK = 'rank'

CATEGORY_RANK = [TRIGGER, LABELED, SPAN_ONLY]
EVENT_TYPE_RANK = [ALCOHOL, DRUG, TOBACCO, EMPLOYMENT, LIVING_STATUS, COVID, SSX]

def trig_first(df):
    
    trig_tmp = '1{}'.format(TRIGGER)
    map1 = lambda x: trig_tmp if x == TRIGGER  else x
    map2 = lambda x: TRIGGER  if x == trig_tmp else x
    df[ARGUMENT] = df[ARGUMENT].map(map1)  
    df = df.sort_values([EVENT_TYPE, ARGUMENT], ascending=True)
    df[ARGUMENT] = df[ARGUMENT].map(map2)      

    return df

def PRF(df):

    df['P'] = df['TP']/(df['NP'].astype(float))
    df['R'] = df['TP']/(df['NT'].astype(float))
    df['F1'] = 2*df['P']*df['R']/(df['P'] + df['R'])
    
    return df


def to_list(X):

    if (X is None) or isinstance(X, list):
        return X
    else:
        return [X]

class Event(object):
    
    def __init__(self, type_, arguments=None):
        '''
        args:
            type_: type of event as string
            arguments: ordered dictionary of arguments, where key is span type
            trigger_type: name of trigger type as string
            status_type: name of status type as string
            
        '''
        self.type_ = type_
        
        
        self.trig_type = TRIGGER
        self.status_type = STATUS
        self.entity_type = ENTITY


        self.arguments = [] if arguments is None else arguments
        self.check_duplicates()

    def __str__(self):
        '''
        String representation of Event
        '''
        out = []
        out.append('Event(type_={}'.format(self.type_))
        for argument in self.arguments:
            out.append('\t{}'.format(argument))
        out[-1]+= ')'
        return "\n".join(out)

    def arg_types(self):
        '''
        Get list of argument types
        '''
        return [arg.type_ for arg in self.arguments]
    
    def check_duplicates(self):
        '''
        Make sure there is only 1 trigger and 1 status span
        '''
        
        # List of argument types
        arg_types = self.arg_types()
        
        # Counts trigger and status spans/arguments
        trig_count = arg_types.count(self.trig_type)
        status_count = arg_types.count(self.status_type)

        assert (trig_count <= 1) and (status_count <= 1), '\n{}'.format(str(self))
        
        return True 
    
    def add_argument(self, argument):        
        '''
        Add argument
        '''
        self.arguments.append(argument)
        self.check_duplicates()        
        return True

    def get_arg(self, type_):
        '''
        Get arguments of specified type
        '''
        args = [arg for arg in self.arguments if arg.type_ == type_]
        
        n_args = len(args)
        
        if n_args == 0:
            return None
        elif n_args == 1:
            return args[0]
        else:
            raise ValueError("Too many args")
        


    def get_arg_types(self, type_):
        '''
        Get arguments of specified type
        '''
        return [arg for arg in self.arguments if arg.type_ == type_]

    def trigger(self):
        '''
        Get trigger argument
        '''
        args = self.get_arg_types(self.trig_type)

        if len(args) == 0:
            return None
        elif len(args) == 1:
            return args[0]
        else:   
            ValueError("Multiple triggers: {}".format(self))       

    def status(self):
        '''
        Get status argument
        '''
        args = self.get_arg_types(self.status_type)
               
        return args[0] if len(args) == 1 else None

    def entities(self):
        '''
        Get entity arguments
        '''
        return self.get_arg_types(self.entity_type)

        
    def has_trigger(self):
        '''
        Boolean indicating whether event has trigger
        
        Used for events consisting of arguments not located in same 
        sentences trigger
        '''
        return self.trigger() is not None

    def max_tok_idx(self):
        '''
        Get maximum token indices across spans
        '''
        return max([s.tok_idxs[1] for s in self.arguments])

    
    def to_dict(self, char_indices=None):
        
        event_dict = {}
        event_dict['type'] = self.type_
        event_dict['arguments'] = []
        for arg in self.arguments:
            arg_dict = arg.to_dict(char_indices=char_indices)
            event_dict['arguments'].append(arg_dict)
        
        return event_dict


class EventScorer(object):
    
    def __init__(self, exclude = None,
                true_spans_incl = False,
                pred_spans_incl = False,
                by_doc = False,
                partial_only = False,
                score_type = None
                ):
        '''
        
        
        Parameters
        ----------
        exclude: list of tuple (column name, value) to exclude 
                    e.g. [('argument', 'degree')]
        true_spans_sep: Boolean indicating whether events and spans 
                        are provided in 'true' in fit
        pred_spans_sep: Boolean indicating whether events and spans 
                        are provided in 'pred' in fit            
        by_doc: indicates whether 'true' and 'pred' in fit are provided 
                        at document level, e.g. [docs [sentences [events]]] 
                        if false, assumed to be of the form [sentences [events]]
        
        '''     
        
        self.exclude = exclude
        self.true_spans_incl = true_spans_incl
        self.pred_spans_incl = pred_spans_incl
        self.by_doc = by_doc
        self.partial_only = partial_only
        self.score_type = score_type


    def process_y(self, y, spans_incl):
        '''
        Process input, y, to get spans and events
        '''

        # If provided as list of documents, flatten, so list of sentences
        flatten = lambda docs: [sent for doc in docs for sent in doc]    
            
        # Spans provided, along with events
        if spans_incl:

            # Separate spans and events
            
            # spans and events packaged as tuple of list
            if isinstance(y, tuple):            
                spans, events = y

            # spans and events packaged as list of tuple
            else:

                spans, events = zip(*y)
                spans = list(spans)
                events = list(events)

            
            # Flatten spans and events
            if self.by_doc:
                spans = flatten(spans)
                events = flatten(events)
        
        # Only events provided
        else:
            
            # Flatten events
            if self.by_doc:
                y = flatten(y)
            
            # Get events and spans
            events = y        
            spans = get_spans_from_events(y)

        assert len(spans) == len(events)
            
        return (spans, events)         
        
    def fit(self, true, pred, params=None, score_type=PARTIAL):
        '''        
        Score predictions
        '''
        
        
        # This is a temporary patch to maintain backwards compatibility
        if not hasattr(self, 'score_type'):
            self.score_type = None
        

        # Get spans and events
        spans_true, events_true = self.process_y(true, self.true_spans_incl) 
        spans_pred, events_pred = self.process_y(pred, self.pred_spans_incl) 
        
        # Check sentence count
        assert len(spans_true) == len(spans_pred), "length mismatch"
        assert len(events_true) == len(events_pred), "length mismatch"
        
        # Evaluation/match configurations
        if self.score_type is not None:

            # Get counts
            df = event_eval( \
                        spans_true = spans_true,
                        events_true = events_true, 
                        spans_pred = spans_pred,
                        events_pred = events_pred,
                        score_type = self.score_type, 
                        exclude = self.exclude)
            
            # Include each parameter in data frame       
            if params is not None:
                df = add_params_to_df(df, params)        

            return df
            
        
        elif self.partial_only:
            
            # Get counts
            df = event_eval( \
                        spans_true = spans_true,
                        events_true = events_true, 
                        spans_pred = spans_pred,
                        events_pred = events_pred,
                        score_type = score_type, 
                        exclude = self.exclude)
            
            # Include each parameter in data frame       
            if params is not None:
                df = add_params_to_df(df, params)        

            return df
            
                   
        else:
            
            cases = [ \
                {'name': 'partial', 'score_type': PARTIAL},
                {'name': 'exact',   'score_type': EXACT},
                {'name': 'wo_det',   'score_type': NO_DET},
                {'name': 'partial', 'score_type': PARTIAL},
                {'name': 'any_overlap', 'score_type': ANY_OVERLAP},
            ]
            
            # Loop on evaluation configurations
            dfs = OrderedDict()
            for case in cases:
                       
                # Get counts
                df = event_eval( \
                            spans_true = spans_true,
                            events_true = events_true, 
                            spans_pred = spans_pred,
                            events_pred = events_pred,
                            score_type = case['score_type'], 
                            exclude = self.exclude)
                

                dfs['{}_{}'.format(case['name'], 'Full')] = df

                combos = [ \
                        (ARGUMENT,  SPAN), 
                        (LABEL, SPAN), 
                        (EVENT_TYPE,  ARGUMENT),
                        (EVAL_TYPE, None)]
                
                for col, evl in combos:
                    if evl is not None:
                        df_ = df[df[EVAL_TYPE] == evl]
                    else:
                        df_ = df.copy()
                    
                    df_ = df_.groupby(col).sum()
                    df_ = score_df(df_)
                    dfs['{}_{}'.format(case['name'], col)] = df_
                
            # Include each parameter in data frame       
            if params is not None:
                for k, df in dfs.items():
                    dfs[k] = add_params_to_df(df, params)        
            
            if self.partial_only:
                return dfs['{}_{}'.format('partial', 'Full')]
            else:
                return dfs




class Relation(object):
    
    def __init__(self, trigger, argument):
        
        
        self.trigger = trigger
        self.argument = argument

    def __str__(self):
        return 'Relation(\n\ttrigger={}, \n\targument={})'.format(self.trigger, self.argument)
       
class Span(object):        
    
    
    def __init__(self, type_, sent_idx, tok_idxs, \
                tokens=None, label=None, char_idxs=None, text=None):
    
        self.type_ = type_
        self.sent_idx = sent_idx
        self.tok_idxs = tok_idxs
        self.tokens = tokens    
        self.label = label
        self.char_idxs = char_idxs
        self.text = text
        

    def _str(self):
        return 'Span(type_={}, sent_idx={}, tok_idxs={}, tokens={}, label={}, char_idxs={}, text={})'.format( \
                self.type_, self.sent_idx, self.tok_idxs, self.tokens, self.label, self.char_idxs, self.text)

    def __str__(self):
        return self._str()

    def __repr__(self):
        return self._str()

    def __eq__(self, other):
        return (self.type_ == other.type_) and \
               (self.sent_idx == other.sent_idx) and \
               (self.tok_idxs == other.tok_idxs) and \
               (self.label == other.label)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        tokens = None if self.tokens is None else tuple(self.tokens)
        return hash((self.type_, self.sent_idx, self.tok_idxs, tokens, self.label))

    def to_dict(self, char_indices=None):
        
        d = {}
        d['type'] = self.type_
        d['label'] = self.label
        
        
        if char_indices is not None:
            self.add_char_idxs(char_indices)
        
        d['text'] = self.text
        d['indices'] = self.char_idxs
        
        return d

    def compare_type(self, other):
        '''
        Compare type with other Span
        '''
        assert isinstance(other, Span)
        return self.type_ == other.type_

    def compare_label(self, other):
        '''
        Compare label with other Span
        '''
        assert isinstance(other, Span)
        return self.label == other.label

    def compare_sent(self, other):
        '''
        Compare sentence indices with other Span
        '''
        assert isinstance(other, Span)        
        return self.sent_idx == other.sent_idx

    def compare_indices(self, other, eval_type=EXACT):
        '''
        Compare span indices


        Parameters
        ----------
        other: Span for comparison
                
        eval_type: str indicating evaluation type:
              'exact': indices equivalent if indices are exactly equal AND sentence indices match
              'partial': indices equivalent if indices overlap AND sentence indices match
        
        Returns
        -------
        output: integer score, where:
            if eval_type in [EXACT, NO_DET, ANY_OVERLAP]:
                output = 0 (not equal) or 1 (equal)
            if eval_type in [PARTIAL]:
                output = number of overlapping tokens
                 
        
        '''
        
        # Check types
        assert isinstance(other, Span)
        assert isinstance(eval_type, str)
        
        
        
        # Sentence match
        sent_match = self.compare_sent(other)

        # Sentences dont match
        if not sent_match:
            output = 0
        
        # Exact match
        elif eval_type == EXACT:
            output = int(self.tok_idxs == other.tok_idxs)

        # Exact match, neglecting determiners
        elif eval_type == NO_DET:
            
            def adj_start(span):
                tok_idxs = span.tok_idxs
                if span.tokens[0].lower() in IGNORE_TOKENS:
                    tok_idxs = (tok_idxs[0]+1, tok_idxs[1])
                return tok_idxs
            
           
            self_tok_idxs = adj_start(self)               
            other_tok_idxs = adj_start(other)
                
            output = int(self_tok_idxs == other_tok_idxs)

        # Token-level match count                
        elif eval_type == PARTIAL:
            output = len(get_overlap(self.tok_idxs, other.tok_idxs))
            
        # Any overlap
        elif eval_type == ANY_OVERLAP:
            output = int(is_overlap(self.tok_idxs, other.tok_idxs))
        
        else:
            raise ValueError("Invalid eval_type: {}".format(eval_type))

        return output

    def compare(self, other, eval_type=EXACT):
        '''
        Compare spans

        Parameters
        ----------'s
        other: Span for comparison

        eval_type: str indicating evaluation type:
              'exact':  spans equiv. if (indices equal) and (type equal) and (label equal)
              'partial':  spans equiv. if (indices overlap) and (type equal) and (label equal)
              'label': for spans with positive (non-None) labels, spans equiv. if (type equal) and (label equal) - don't care about indices
                       for spans with None label, spans equiv. if (type equal) and (indices exactly equal) - don't care about label



        Returns
        -------
        
        
        
        '''
        
        if other is None:
            return False
        
        # Check type        
        assert isinstance(other, Span)        

        # Is trigger?
        is_trigger = (self.type_ == TRIGGER) and \
                     (other.type_ == TRIGGER)

        # Is span-only argument?
        is_span_only = (self.label is None) and \
                       (other.label is None)
                       
        # Is labeled argument?                       
        is_labeled = not is_span_only
               
        # Compare type
        type_match = self.compare_type(other)

        # Compare label
        label_match = self.compare_label(other)

        # Evaluation type 
        if eval_type == SSX:

            # Trigger
            if is_trigger:
                indices_match = self.compare_indices(other, EXACT)
            
            # Span-only argument
            elif is_span_only:
                indices_match = self.compare_indices(other, PARTIAL)

            # Labeled argument
            elif is_labeled:
                indices_match = True
                
            # Error catch
            else:
                raise ValueError("could not determine indices_match")



        # Compare token indices
        elif eval_type in [EXACT, NO_DET]: #, PARTIAL]:
            indices_match = self.compare_indices(other, eval_type)
        
        # Compare indices, if labels None
        elif eval_type in [PARTIAL, ANY_OVERLAP]:
            
            # Span-only Argument
            # Consider indices, because no labels
            if is_span_only:
                indices_match = \
                       self.compare_indices(other, eval_type)

            # Labeled Argument
            # Ignore indices
            else:
                indices_match = True            
        
        # Invalid evaluation type
        else:
            raise ValueError("Invalid eval_type: {}".format(eval_type))
    
        
        
        return type_match and label_match and indices_match

    def add_char_idxs(self, indices):
        '''
        Include character indices for token spans
        '''

        # Sentence index and token indices
        sent_idx = self.sent_idx
        tok_start, tok_end = self.tok_idxs
        
        # Associated character indices
        char_idxs = indices[sent_idx][tok_start:tok_end]            
        char_start = char_idxs[0][0]
        char_end = char_idxs[-1][1]
        
        # Get character indices of start of each token
        token_starts = [start - char_start for start, end in char_idxs]
                
        # Convert tokens to string
        text = '' 
        for start, tok in zip(token_starts, self.tokens):
            text += (start - len(text))*' '
            text += tok
        text = "".join(text)

        self.char_idxs = (char_start, char_end)
        self.text = text
        
        return True
        



def get_overlap(indices1, indices2):
    
    if (indices1 is None) or (indices2 is None):
        return []
    
    gen_idx = lambda x: list(range(*x))
    
    x1 = gen_idx(indices1)
    x2 = gen_idx(indices2)
    overlap = [v for v in x1 if v in x2]
 
    return overlap


def is_overlap(indices1, indices2):
    '''
    Determine if span indices overlap   
    '''
    
    #if (indices1 is None) or (indices2 is None):
    #    return False    
    #else:    
    #
    #    start1, end1 = indices1 
    #    start2, end2 = indices2
    #
    #    # Decrement because end indices are exclusive
    #    end1 += -1
    #    end2 += -1    
    #    
    #    return (start1 <= end2) and (start2 <= end1)
    return len(get_overlap(indices1, indices2)) > 0


def BIO_to_span_seq(seq, check_count=False, as_Span=True):
    '''
    
    Finds spans in BIO sequence
    
    NOTE: start span index is inclusive, end span index is exclusive
            e.g. like Python lists
    
    '''

    # Not in a span
    in_span = False
    

    spans = []
    b_count = 0
    start = -1
    end = -1
    active_tag = None
    for i_tok, x in enumerate(seq):
        
        # Current tag
        tag = strip_BIO_tok(x)
        
        # Outside token
        if is_outside(x):
           
            # The span has ended
            if active_tag is not None:
                s = Span( \
                    type_ = active_tag, 
                    sent_idx = None, 
                    tok_idxs = (start, end))
                spans.append(s)

            # Not in a span
            active_tag = None
        
        # Span beginning
        elif is_begin(x):
            
            # The span has ended
            if active_tag is not None:
                s = Span( \
                    type_ = active_tag, 
                    sent_idx = None, 
                    tok_idxs = (start, end))
                spans.append(s)

            # Update active tag
            active_tag = tag
            
            # Index of current span start
            start = i_tok
            end = i_tok + 1
                        
            # Increment b count
            b_count += 1
        
        # Span inside and current tag matches active tag
        # e.g. well-formed span
        elif is_inside(x) and (tag == active_tag):
            
            end += 1
            
        # This is an inside span that is ill formed
        else:
            
            # Append ill formed span
            if active_tag is not None:
                s = Span( \
                    type_ = active_tag, 
                    sent_idx = None, 
                    tok_idxs = (start, end))
                spans.append(s)
            # Update active tag
            active_tag = tag

            # Index of current span start
            start = i_tok
            end = i_tok

    # Last token might be part of a valid span
    if active_tag is not None:
        s = Span( \
            type_ = active_tag, 
            sent_idx = None, 
            tok_idxs = (start, end))
        spans.append(s)        

    # Get span count       
    s_count = len(spans)
   
    if check_count and (b_count != s_count):
        msg = \
        '''Count mismatch:
        seq = {}
        Begin count = {}
        span count = {}'''.format(seq, b_count, s_count)
        logging.warn(msg)
    
    if as_Span:
        return spans
    else:
        types_ = [s.type_ for s in spans]
        indices = [list(s.indices) for s in spans]
        return (types_, indices)

def BIO_to_span_doc(doc):

    return [BIO_to_span_seq(seq) for seq in doc]




def get_spans_from_events(events):
    '''
    Get spans from events
    '''
    
    # Initialize output 
    n_sent = len(events)        
    spans = [[] for _ in range(n_sent)]        

    # Iterate over sentences
    for i_sent, sent in enumerate(events):
        
        # Iterate over events in current sentence
        for evt in sent:
            
            # Get trigger sentence index
            trig = evt.trigger()
            i_trig = None if trig is None else trig.sent_idx
            
            # Iterate over arguments in current event
            for span in evt.arguments:
                
                # Sentence index for current span
                i_span = span.sent_idx
                
                # No trigger so cannot adjust
                # All gold standard events have a trigger, so this only 
                # impacts poorly predicted events
                if i_trig is None:
                    i_span_adj = i_sent
                
                # Adjusted sentence index
                else:
                    i_span_adj = i_sent + (i_span - i_trig)
                
                spans[i_span_adj].append(span)
                
    return spans
                       


def unique_types(X):
    '''
    Get set of unique types
    '''
    return set([x.type_ for x in X])

    
def type_filt(X, type_): 
    '''
    Filter based on type
    '''
    return [x for x in X if x.type_ == type_]

    

def align_spans(spans_true, spans_pred, attr='type_'):
    '''
    Align two lists of spans
    
    Parameters
    ----------
    spans_true: list of Span
    spans_pred: list of Span
    attr: align span with same attribute, as defined by 'attr'
            For trigger spans, this should be 'label'. 
             For all other spans this should be 'type_'

    align_type: 

    Returns
    -------
    
    '''    

    # Get span types from list of spans
    def get_types(spans, attr=attr): 
        return [getattr(s, attr) for s in spans if s is not None]
    
    # Convert list of span to order dictionary with index
    def as_dict(spans):
        return OrderedDict([(i, s) for i, s in enumerate(spans) \
                                                     if s is not None])
      
    # Filter list of spans by type
    def get_subset(span_dict, attr_val, attr=attr): 
        return [k for k, s in span_dict.items() \
                      if getattr(s, attr) == attr_val]
          
    # Get midpoint of token indices, assuming end idx is exclusive
    get_mid = lambda s:  np.mean(np.array(s) - np.array([0, 1]))
        
    
    # Get span types
    span_types = list(set(get_types(spans_true) + \
                          get_types(spans_pred)))

    # Convert to order dictionary with index
    T = as_dict(spans_true)
    P = as_dict(spans_pred)

    # Aligned spans
    aligned = []
        
    # Loop on spans types
    for typ in span_types:
        
        I = get_subset(T, attr_val=typ)
        J = get_subset(P, attr_val=typ)
            
        # All combinations between true and predicted spans
        combos = [(i, j) for i in I for j in J]

        # Get distance between spans
        dist = []
        for i, j in combos:
            t = T[i]
            p = P[j]
            
            xi = get_mid(t.tok_idxs)
            xj = get_mid(p.tok_idxs)
            
            # Distance between midpoints
            d = np.abs(xi - xj)
            
            s = '{} - {}'.format(str(t), str(p))
            
            dist.append((i, j, d, s))
        dist.sort(key=lambda tup: (tup[2], tup[3]))

        # Iterate while all combinations exist
        while len(dist) > 0:
    
            # Closest spans
            i, j, d, s = dist.pop(0)
            aligned.append((i, j))
        
            # Removed consumes spans from list of combinations
            dist_tmp = []            
            for i_, j_, d_, s_, in dist:
                if (i_ != i) and (j_ != j):
                    dist_tmp.append((i_, j_, d_, s_))
            dist = dist_tmp        

    return aligned





def align_events(events_true, events_pred): 
    '''
    Aligned to lists of events (events in a sentence)
    
    Parameters
    ----------
    events_true: list of Event
    events_pred: list of Event
    align_type: alignment type as string
    
    
    Returns
    -------
    alignment indices as list of tuple [(true idx, pred idx),..], 
                                       e.g. [(2,4), (4,5),...]
    
    '''

    # Trigger span indices for current sentence
    get_trig = lambda events: [evt.trigger() for evt in events]
    trig_true = get_trig(events_true)
    trig_pred = get_trig(events_pred)

    # Align triggers
    return align_spans(trig_true, trig_pred, attr='label')


    


def index_tuple(event_type, arg_type, arg_label):
    
    if event_type is None:
        event_type = NONE_PH
    if arg_label is None:
        arg_label = NONE_PH
    return (event_type, arg_type, arg_label)


        
def count_spans(spans, unique_only=False, score_type=None):
    '''
    Count spans
    
    Parameter
    ---------
    events: Event, list of list of Span (i.e. Sentence)
    '''
    
        
    # Loop on sentences
    spans_flat = []
    for sent_spans in spans:
        s = list(set(sent_spans)) if unique_only else sent_spans
        spans_flat.extend(s)
        
    # Tuple representation, span type and label
    counter = Counter()
    for span in spans_flat:
        
        # Count ({0,1} for span-level OR 0+ for token-level)    
        n = span.compare(span, score_type)
        
        # Key
        k = index_tuple(None, span.type_, span.label)        
        counter[k] += n
    
    return counter


def count_event_spans(events, score_type):
    '''
    Count spans
    
    Parameter
    ---------
    events: Event, list of Event (i.e. Sentence), or 
            list of list of Event (i.e. document)
    '''

    # Iterate over sentences
    counter = Counter()
    for sent in events:
        # Iterate over events in current sentence
        for evt in sent:             
            # Iterate over arguments in current event
            for span in evt.arguments:
        
                # Count ({0,1} for span-level OR 0+ for token-level)    
                n = span.compare(span, score_type)
                
                # Key
                k = index_tuple(evt.type_, span.type_, span.label)
                counter[k] += n    
           
    return counter




def compare_spans(spans_true, spans_pred, score_type, unique_only=False):
    
    '''
    Align and compare spans (no consideration for event assignment)
    
    Parameters
    ----------
    spans_true: list of Span
    spans_pred: list of Span


    score_type: scoring type as str
    
    Returns
    -------
    counts: Counter of true positives    
    '''                    

    # Filter for uniqueness
    if unique_only:
        spans_true = list(set(spans_true))
        spans_pred = list(set(spans_pred))

    # Align spans in sentence
    alignment = align_spans(spans_true, spans_pred) 
    
    # Loop on aligned events in current sentence
    counter = Counter()
    for i, j in alignment:
        
        t = spans_true[i]
        p = spans_pred[j]
        
        # Spans match
        score = t.compare(p, eval_type=score_type)
        counter[index_tuple(None, t.type_, t.label)] += score

    return counter
    
def compare_spans_doc(spans_true, spans_pred, score_type,
                                            unique_only=False):
    '''
    Align and compare spans (no consideration for event assignment)
    
    
    Parameters
    ----------
    spans_true: list of list of Span
    spans_pred: list of list of Span

    
    score_type: scoring type as str 
    
    Returns
    -------
    counts: Counter of true positives
    '''        
    
    assert len(spans_true) == len(spans_pred)            

    # Loop on sentences
    counts = Counter()
    for spans_t, spans_p in zip(spans_true, spans_pred):
        counts += compare_spans( \
                                spans_true = spans_t, 
                                spans_pred = spans_p, 
                                score_type = score_type, 
                                unique_only = unique_only)

    return counts

    
def compare_events(event_true, event_pred, score_type):
    '''
    Compare two events and count true positives
    

    Parameters
    ----------
    score_type: scoring type as str 
    '''

    # Get trigger
    trig_true = event_true.trigger()
    trig_pred = event_pred.trigger()

    if score_type == PARTIAL:
        assert trig_true.compare(trig_pred, score_type), "triggers don't match"
    
    # Check trigger
    if trig_true.compare(trig_pred, score_type):
        to_dict = lambda evt: {arg.type_:arg for arg in evt.arguments}
        
        args_true = to_dict(event_true)
        args_pred = to_dict(event_pred)
        
        common_args = list(set(args_true.keys()) & \
                           set(args_pred.keys()))
    
        counter = Counter()    
        for arg in common_args:
            span_true = args_true[arg]
            span_pred = args_pred[arg]
            
            score = span_true.compare(span_pred, eval_type=score_type)
            k = index_tuple(event_true.type_, span_true.type_, span_true.label)
            counter[k] += score
            
        return counter
        

    else:

        return Counter()



def compare_events_sent(events_true, events_pred, score_type):
    '''
    Compare events within sentence

    Parameters
    ----------
    score_type: scoring type as str 
              
    '''

    # Align triggers
    alignment = align_events(events_true, events_pred)

    # Loop on aligned events in current sentence
    counts = Counter()
    for i, j in alignment:
        counts += compare_events( \
                            event_true = events_true[i], 
                            event_pred = events_pred[j],
                            score_type = score_type)
     
    return counts


def compare_events_doc(events_true, events_pred, score_type):
    '''
    Compare events for sentences in document

    Parameters
    ----------
    score_type: scoring type as str 
    '''
    
    assert len(events_true) == len(events_pred)
    
    # Loop on sentences
    counts = Counter()
    for t, p in zip(events_true, events_pred):
        counts += compare_events_sent(t, p, score_type = score_type)

    return counts


def score_df(df):


    # Calculate precision, recall, and F1
    df[FN] = df[NT] - df[TP]
    df[FP] = df[NP] - df[TP]
    df[P] = df[TP]/df[NP]
    df[R] = df[TP]/df[NT]
    df[F1] = 2*(df[P]*df[R])/(df[P]+df[R])
    
    return df
    
def event_eval(spans_true, events_true, spans_pred, events_pred, \
                score_type, exclude=None):
    '''
    Align and compare events
    '''          
    unique_only = True
    
    assert len(events_true) == len(events_pred)
    assert len(spans_true) == len(spans_pred)
    assert len(spans_true) == len(events_pred)

    # Span counts in truth and prediction
    span_nt = count_spans(spans_true, \
                                    unique_only = unique_only, 
                                    score_type = score_type)
    
    span_np = count_spans(spans_pred, \
                                    unique_only = unique_only, 
                                    score_type = score_type)

    # Span true positives
    span_tp = compare_spans_doc(spans_true, spans_pred, \
                                     score_type = score_type, 
                                     unique_only = unique_only)

    # Force all missing counter keys to have count of 0
    for k in list(span_nt.keys()) + list(span_np.keys()):
        span_nt[k] += 0
        span_np[k] += 0
        span_tp[k] += 0
                   
    # Argument counts in truth and prediction
    arg_nt = count_event_spans(events_true, score_type=score_type)
    arg_np = count_event_spans(events_pred, score_type=score_type)
    
    # Argument true positives
    arg_tp = compare_events_doc(events_true, events_pred, \
                                score_type = score_type)

    # Force all missing counter keys to have count of 0
    for k in list(arg_nt.keys()) + list(arg_np.keys()):
        arg_nt[k] += 0
        arg_np[k] += 0
        arg_tp[k] += 0

    # Columns
    cols = [NT, NP, TP]

    # Convert span count to data frame
    series = [span_nt, span_np, span_tp]
    series = [{(SPAN,)+k:c for k, c in s.items()} for s in series]
    series = [pd.Series(s) for s in series]
    span_df = pd.concat(series, axis=1)
    span_df.columns = cols
    
    # Convert argument counts to data frame
    series = [arg_nt, arg_np, arg_tp]
    series = [{(ARGUMENT,)+k:c for k, c in s.items()} for s in series]
    series = [pd.Series(s) for s in series]
    arg_df = pd.concat(series, axis=1)
    arg_df.columns = cols    
    
    # Concatenate data frames
    df = pd.concat([span_df, arg_df])

    if len(df) == 0:
         columns = ['Evaluation type',  'Event type', 'Argument', 'Label',
                    'NT', 'NP', 'TP', 'FN', 'FP', 'P', 'R', 'F1']
         df_empty = pd.DataFrame({c : [] for c in columns})
         return df_empty
    
    else:
        # Make index column
        df[INDEX_COLS] = pd.DataFrame(df.index.tolist(), index=df.index)
        df = df.loc[:, INDEX_COLS + cols].reset_index(drop=True)
        
        # Merge with the original dataframe and sort
        df = df.sort_values(by=INDEX_COLS, ascending=False)
        df = score_df(df)

        if exclude is not None:
            df = filter_df(df, exclude=exclude)
        
        return df


def events2spans(events):
    '''
    Convert events to spans
    
    Parameters
    ----------
    events: document labels as sequence of sentences as sequence of Event
            e.g. [[Event(), ... Event()], 
                  [Event(), ... Event()],
                  [Event(), ... Event()]]
    
    Returns
    -------
    evt_spans: document labels as sequence of sentences as sequence of Span
            e.g. [[(event_type, Span()), ... (event_type, Span())],
                  [(event_type, Span()), ... (event_type, Span())],
                  [(event_type, Span()), ... (event_type, Span())]]
    '''
    
    assert isinstance(events, list)
    assert isinstance(events[0], list)
    if len(events[0]) > 0:
        assert isinstance(events[0][0], Event)
    
    # Number of sentences in doc
    sent_cnt = len(events)

    # Original span count   
    cnt1 = sum([len(evt.arguments) for sent in events for evt in sent])

    # Initialize output
    evt_spans = [[] for _ in range(sent_cnt)]

    # Loop on sentences in document
    for sent in events:
     
        # Loop on events in current sentence
        for evt in sent:

            # Loop on spans in current event
            for span in evt.arguments:
                
                # Incorporate event type and span
                evt_spans[span.sent_idx].append((evt.type_, span))

    # New span count   
    cnt2 = sum([len(sent) for sent in evt_spans])
    assert cnt1 == cnt2, '{} vs {}'.format(span1_cnt, cnt2)            

    return evt_spans         



def events2sent_labs(events, label_map, arg_type, label_to_id=None):
    '''
    Get sentence-level labels from events
    Parameters
    ----------
    events: document labels as sequence of sentences as sequence of Event
            e.g. [[Event(), ... Event()], 
                  [Event(), ... Event()],
                  [Event(), ... Event()]]
    
    label_map: event types with label mapping (hierarchy) 
                as dict of list
            e.g. {'Alcohol': [outside, past, current], 
                  'Drug': [outside, past, current], 
                  'Tobacco': [outside, past, current]}

    arg_type: argument type of interest as string
            e.g. 'Trigger' or 'Status'    
                    
    
    Returns
    -------
    labels: sentence-level labels as sequence of dict of labels
            e.g. [{Alcohol: Outside, Tobacco: Tobacco},
                  {Alcohol: Alcohol, Tobacco: Tobacco}]
                or
                [{Alcohol: current, Tobacco: none},
                 {Alcohol: past, Tobacco: none}]
    '''    

    # Initialize with negative label
    lab_init = OrderedDict()
    for evt_typ, labs in label_map.items():
        lab_init[evt_typ] = [labs[0]]

    # Loop on sentences in document
    labels = []
    for sent_evts in events:
        
        # Initialize sentence-level labels
        d = copy.deepcopy(lab_init)
        
        # Loop on events in current sentence
        for evt in sent_evts:
            
            # Loop on arguments in current event
            for arg in evt.arguments:
                
                # Event and argument type match, so include
                if (evt.type_ in label_map) and (arg.type_ == arg_type):
                    d[evt.type_].append(arg.label)
        
        # Process aggregated labels        
        for evt_typ, labs in d.items():
            
            # Merge labels for each event type
            lab = merge_labels_(labs, hierarchy=label_map[evt_typ])

            # Apply map to labels
            if label_to_id is not None:
                lab = label_to_id[evt_typ][lab]

            d[evt_typ] = lab

        # Append sentence label representation
        labels.append(d)   

    
    return labels



def sent_labs2ids(labels, label_map):
    '''
    Get label IDs from sentence-level labels
    
    Parameters
    ----------
    labels: sentence-level labels as sequence of dict of labels
            e.g. [{Alcohol: Outside, Tobacco: Tobacco},
                  {Alcohol: Alcohol, Tobacco: Tobacco}]
                or
                [{Alcohol: current, Tobacco: none},
                 {Alcohol: past, Tobacco: none}]
    
    label_map: event types with label mapping (hierarchy) 
                as dict of list
            e.g. {'Alcohol': [outside, past, current], 
                  'Drug': [outside, past, current], 
                  'Tobacco': [outside, past, current]}
   
    Returns
    -------
    ids:  sentence-level labels as sequence of dict of labels
            e.g. [{Alcohol: 0, Tobacco: 1},
                  {Alcohol: 0, Tobacco: 1}]
                or
                [{Alcohol: 3, Tobacco: 1},
                 {Alcohol: 2, Tobacco: 1}]
    
    '''        
    
    # Get label mapping dictionaries
    lab2id, _ = get_mapping_dicts(label_map)
    
    # Loop on sentences
    ids = []
    for sent in labels:
        
        # Map label for each event type
        ids_dict = OrderedDict()
        for evt_typ, lab in sent.items():
            ids_dict[evt_typ] = lab2id[evt_typ][lab]
        
        ids.append(ids_dict)

    return ids
    
def events2seq_tags(events, label_map, max_len, pad_start, \
        include_prefix=True, label_to_id=None):
    '''
    Get sequence tag labels from events
    
    Parameters
    ----------
    events: document labels as sequence of sentences as sequence of Event
            e.g. [[Event(), ... Event()], 
                  [Event(), ... Event()],
                  [Event(), ... Event()]]

    label_map: event type-argument type combinations as dict of list 
            e.g. {'Alcohol': ['Amount', 'Frequency'], 
                  'Drug': ['Amount', 'Frequency'], 
                  'Tobacco': ['Amount', 'Frequency']}
    
    max_len: maximum sequence length as int

    pad_start: Boolean indicating whether start of sequence is padded 
                with start of sequence token

    
    Returns
    -------
    

    '''    

    # Assume negative label is first label in hierarchy    
    neg_labels = OrderedDict([(evt_typ, labs[0]) \
                                for evt_typ, labs in label_map.items()])

    # Convert events to spans
    evt_spans = events2spans(events)

    # Initialize output
    labels = []

    # Loop on sentences in document
    for sent in evt_spans:
        
        # Initialize
        d_labels = OrderedDict()
        for evt_typ, neg_lab in neg_labels.items():
            
            if include_prefix:
                d_labels[evt_typ] = [(OUTSIDE, neg_lab) for _ in range(max_len)]
            else: 
                d_labels[evt_typ] = [neg_lab for _ in range(max_len)]
        
        # Loop on spans in current sentence
        for evt_typ, span in sent:
            
            # Event type and argument type match
            if (evt_typ in label_map) and \
                (span.type_ in label_map[evt_typ]):

                # Token indices of span                    
                start, end = span.tok_idxs
                
                # Loop on tokens in span                
                for j, tok_idx in enumerate(range(start, end)):

                    # Increment index, if padding start
                    tok_idx += int(pad_start) 

                    # Index in range
                    if tok_idx < max_len:
                
                        # Warn if overwriting non-negative labels
                        allowable = [neg_labels[evt_typ], \
                                     span.type_]
                        if include_prefix:
                            x = d_labels[evt_typ][tok_idx][1]
                        else:
                            x = d_labels[evt_typ][tok_idx]
                        if x not in allowable:
                            logging.warn('Converting events to sequence tags: {} not in {}'.format(x, allowable))
            
                        # Prefix and label for current position
                        lab = span.type_
                        if include_prefix:
                            prefix = BEGIN if j == 0 else INSIDE
                            lab = (prefix, lab)
                            
                        # Update current position
                        d_labels[evt_typ][tok_idx] = lab

        # Map label to ID
        if label_to_id is not None:
            for evt_typ, labs in d_labels.items():
                d_labels[evt_typ] = map_1D(labs, label_to_id[evt_typ])

        # Append sentence results
        labels.append(d_labels)

    return labels

def len_check(X, Y):
    assert len(X) == len(Y), '{} vs. {}'.format(len(X), len(Y))


def events2row(events, excl_arg_types=None):
    '''
    Convert a document represented as events to a row
    
    
    Parameters
    ----------
    id_: id of document as string    
    events: list of list of events, e.g. 
                            [[Event, Event, ...], 
                             [Event, Event, ...]]
    
    Returns
    -------
    row: Ordered dictionary containing all event information
    
    '''

    labels = {}
    spans = {}
    
    # Iterate over sentences in document
    for sent_events in events:
        
        # Iterate over event in current sentence
        for event in sent_events:
            
            # Iterate over arguments in current event
            for arg in event.arguments:
                
                arg_typ_incl = (excl_arg_types is None) or \
                               (arg.type_ not in excl_arg_types)
                
                # Label representation
                # Encode labeled spans as one hot encoding
                #   1 = one or more occurrences of category
                if (arg.label is not None) and arg_typ_incl:                    
                    k = (event.type_, arg.type_, arg.label)
                    labels[k] = 1
                
                # Span representation
                # Represent as string
                if arg_typ_incl:              
                    k = (event.type_, arg.type_)
                    if k not in spans:
                        spans[k] = []

                    spans[k].append(' '.join(arg.tokens))
    
    for k in spans:
        spans[k] = '|'.join(spans[k])

    return (labels, spans)
            

def events2table(ids, events, \
                patients=None, 
                path = None, 
                filename ='tabular_predictions.csv',
                excl_arg_types = None):
                    
    '''
    Convert a list of documents represented as events to a table
    '''
    len_check(ids, events)
    
    
    # Iterate over documents
    rows = []
    label_keys = set([])
    span_keys = set([])
    
    assert len(ids) == len(events)
    if patients is None:
        patients = ['unknown']*len(ids)
    else: 
        assert len(ids) == len(patients)
    
    for id_, patient, doc_events in zip(ids, patients, events):
        
        # Row representation of document
        labels, spans = events2row(doc_events, \
                        excl_arg_types = excl_arg_types)
        
        # Accumulate label and span keys
        label_keys = label_keys.union(set(labels.keys()))
        span_keys = span_keys.union(set(spans.keys()))
        
        row = OrderedDict()
        row[('id',)] = id_
        row[('patient',)] = patient
        row.update(labels)
        row.update(spans)
        
        rows.append(row)

    # Create data frame from rows
    df = pd.DataFrame(rows)   
    col_count1 = len(list(df))

    # Sort keys
    triple = lambda tup: (tup[0], tup[1], tup[2])
    label_keys = sorted(list(label_keys), key=triple)         
       
    double = lambda tup: (tup[0], tup[1])
    span_keys = sorted(list(span_keys), key=double)
    
    columns = [('id',), ('patient',)] + label_keys + span_keys
    df = df[columns]
    col_count2 = len(list(df))

    assert col_count1 == col_count2, '{} vs {}'.format(col_count1, col_count2)


    df.rename(columns='_'.join, inplace=True)

    if path is not None:
        fn = os.path.join(path, filename)
        df.to_csv(fn, index=False)
    
    return df    
            

def perf_plot( \
    df,
    path,
    case = None,
    figsize=(2.8, 1.4),
    font_family = 'serif',
    font_type = 'Times New Roman',
    font_size = 6,   
    width = 0.5,
    gap_small = 1.0,
    gap_big = 1.2,
    x_offset = -0.2,
    left=0.15,
    bottom=0.25,
    right=0.98,
    top=0.98,
    ymin = 0.5, # 0.7,
    ymax = 1.0, # 1.04,
    label_columns = True,
    dpi = 800
    ):

    # Font configuration
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.serif'] = [font_type] + plt.rcParams['font.serif']

    logging.info('-'*72)
    logging.info('Event extraction performance plot')
    logging.info('-'*72)
    
    n_df = len(df)
    logging.info('Data frame length:\t{}'.format(n_df))
    
    trig_id = df[(df[EVAL_TYPE] == SPAN) & 
              (df[ARGUMENT] == TRIGGER)]
    n_trig_id = len(trig_id)              
    logging.info('Trigger ID count:\t{}'.format(n_trig_id))
    
    arg_id_labeled = df[(df[EVAL_TYPE] == SPAN) & 
                        (df[ARGUMENT] != TRIGGER) &
                        (df[LABEL] != NONE_PH)]
    n_arg_id_labeled = len(arg_id_labeled)
    logging.info('Argument ID, label count:\t{}'.format(n_arg_id_labeled))
    
    arg_id_span_only = df[(df[EVAL_TYPE] == SPAN) & 
                          (df[ARGUMENT] != TRIGGER) &
                          (df[LABEL] == NONE_PH)]
    n_arg_id_span_only = len(arg_id_span_only)
    logging.info('Argument ID, span-only count:\t{}'.format(n_arg_id_span_only))

    arg_id = df[(df[EVAL_TYPE] == SPAN) & 
                (df[ARGUMENT] != TRIGGER)]
    n_arg_id = len(arg_id)
    logging.info('Argument ID, all count:\t{}'.format(n_arg_id))


    trig_role = df[(df[EVAL_TYPE] == ARGUMENT) & 
              (df[ARGUMENT] == TRIGGER)]
    n_trig_role = len(trig_role)              
    logging.info('Trigger role count:\t{}'.format(n_trig_role))


    arg_role_labeled = df[(df[EVAL_TYPE] == ARGUMENT) & 
                        (df[ARGUMENT] != TRIGGER) &
                        (df[LABEL] != NONE_PH)]
    n_arg_role_labeled = len(arg_role_labeled)
    logging.info('Argument role, labeled count:\t{}'.format(n_arg_role_labeled))

    
    arg_role_span_only = df[(df[EVAL_TYPE] == ARGUMENT) & 
                          (df[ARGUMENT] != TRIGGER) &
                          (df[LABEL] == NONE_PH)]
    n_arg_role_span_only = len(arg_role_span_only)
    logging.info('Argument role, span-only count:\t{}'.format(n_arg_role_span_only))    
    

    arg_role = df[(df[EVAL_TYPE] == ARGUMENT) & 
                (df[ARGUMENT] != TRIGGER)]
    n_arg_role = len(arg_role)
    logging.info('Argument role, all count:\t{}'.format(n_arg_role))
   
    
    assert n_df == n_trig_id + n_arg_id_labeled + n_arg_id_span_only + \
                n_trig_role + n_arg_role_labeled + n_arg_role_span_only

    
    dfs = []
    xticks_mid = []
    xticks_div = []
    xlabel_mid = []
    colors = []
    i = 0
    dfs.append(('', trig_id, i, 'tab:blue'))
    xticks_div.append(i + gap_big/2)

    xticks_mid.append(i)
    xlabel_mid.append('Trigger')

    i += gap_big
    
    i_start = i
    dfs.append(('Labeled', arg_id_labeled, i, 'tab:orange'))
    
    i += gap_small
    dfs.append(('Span-only', arg_id_span_only, i, 'tab:green'))
    
    #i += gap_small
    #dfs.append(('All', arg_id, i, 'tab:blue'))
    

    i_end = i
    xticks_mid.append((i_start + i_end)/2)
    xlabel_mid.append('Argument')
    xticks_div.append(i + gap_big/2)
    
    i += gap_big    
    
    
    i_start = i
    dfs.append(('Labeled', arg_role_labeled, i, 'tab:orange'))
    
    i += gap_small    
    dfs.append(('Span-only', arg_role_span_only, i, 'tab:green'))   


    #i += gap_small
    #dfs.append(('All', arg_role, i, 'tab:blue'))

    i_end = i
    xticks_mid.append((i_start + i_end)/2)
    xlabel_mid.append('Argument Role')    
    
    vals = []
    
    for name, df_, i, c in dfs:
        TP = df_['TP'].sum()
        FP = df_['FP'].sum()
        FN = df_['FN'].sum()
        P = TP/(TP + FP)
        R = TP/(TP + FN)
        F1 = 2*(P*R)/(P+R)
        vals.append((name, F1, i, c))
    
    # Instantiate figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)


    # Iterate over columns
    rects = []
    xticks = []
    xlabels = []
    #ymin = 1
    #ymax = 0
    txt_offset = 0.01
    offset = txt_offset + 0.02
    for j, (name, val, i, c) in enumerate(vals):
    
        # Add bars
        rect = ax.bar([i], [val], width, color=c) #, color=colors[j], label=labels[j])
        rects.append(rect)
        #ymin = min(ymin, val)
        #ymax = max(ymax, val)
        
        if label_columns:
            txt = ax.text(i, val + txt_offset, '{:.2f}'.format(val), color='black', va='bottom', ha='center') #fontweight='bold',


        xticks.append(i)
        xlabels.append(name)

    
    #ymin = np.floor(ymin*20)/20
    #ymax = np.ceil((ymax + offset)*20)/20

    ax.set_ylim((ymin, ymax))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)


    for t, l in zip(xticks_mid, xlabel_mid):   
        ax.text(t, x_offset, l, \
             va='top', ha='center', transform=ax.get_xaxis_transform(), fontweight='bold')

    linewidth = 0.5
    for t in xticks_div:
        line = plt.Line2D([t, t], [x_offset*0.1, x_offset*1.4],
                      transform=ax.get_xaxis_transform(), color='black', linewidth=linewidth)
        line.set_clip_on(False)
        ax.add_line(line)


    ax.set_ylabel('F1', fontweight='bold')
    ax.set_axisbelow(True)    
    ax.yaxis.grid(True)


    # Save figure
    fig = ax.get_figure()
    if os.path.isdir(path):
        for ext in ['pdf', 'png']:
            if case is None:
                 fn = os.path.join(path, 'event_performance.{}'.format(ext))
            else:        
                fn = os.path.join(path, 'event_performance_{}.{}'.format(case, ext))
            
            if ext != 'pdf':
                fig.savefig(fn, dpi=dpi)
            else:
                fig.savefig(fn)

    else:
        fn = path
        fig.savefig(fn, dpi=dpi)

        


    return True


def get_groups(row):
    '''
    Define span-only argument  groups
        e.g. aggregated span-only arguments, everything else separate
    '''
    
    if row[LABEL] == NONE_PH:
        return SPAN_ONLY
    else:
        return row[ARGUMENT]
    
def get_category(row):
    '''
    Defined categories for trigger, span-only arguments, and 
    labeled arguments
    ''' 
    if row[ARGUMENT] == TRIGGER:
        return TRIGGER
    elif row[GROUP] == SPAN_ONLY:
        return SPAN_ONLY
    else:
        return LABELED

def get_rank(row):
    '''
    Get score for sorting rows
    '''
    return CATEGORY_RANK.index(row[CATEGORY])*10 + \
           EVENT_TYPE_RANK.index(row[EVENT_TYPE])


def arg_list(x):
    '''
    Convert entries to argument list
    '''
    return ', '.join(sorted(x.unique().tolist()))


def df_f1(df):
    '''
    Calculate precision, recall, and F1 for data frame
    '''
    df[P] = df[TP]/df[NP]
    df[R] = df[TP]/df[NT]
    df[F1] = 2*(df[P]*df[R])/(df[P]+df[R])
    return df
            
def perf_summary_tables(df, path):


    
    
    '''
    Calculate summary by event
    '''
    
    
    
    # Argument-only results
    if EVAL_TYPE in df:
        df_by_event = df[df[EVAL_TYPE] == ARGUMENT]
    else:        
        df_by_event = df

    # Get span-only groups
    df_by_event[GROUP] = df_by_event.apply(get_groups, axis=1)

    # Aggregate by event type and group
    agg_dict = {NT: 'sum', NP: 'sum', TP: 'sum', FP: 'sum', FN: 'sum', ARGUMENT: arg_list}
    #for c in columns_all:
    #    if c not in agg_dict:
    #        agg_dict[c] = 'first'
            
    df_by_event = df_by_event.groupby([EVENT_TYPE, GROUP], as_index=False).agg(agg_dict)

    # Calculate metrics
    df_by_event = df_f1(df_by_event)
    
    # Define broader categories and sort
    df_by_event[CATEGORY] = df_by_event.apply(get_category, axis=1)
    df_by_event[RANK] = df_by_event.apply(get_rank, axis=1)
    df_by_event = df_by_event.sort_values([RANK, ARGUMENT])

    # Restrict and reorder columns
    df_by_event = df_by_event[[CATEGORY, EVENT_TYPE, ARGUMENT, NT, P, R, F1]]
    
    '''
    Calculate summary by trigger, labeled arguments, 
    and span-only arguments
    '''
    # Argument-only results
    if EVAL_TYPE in df:
        df_summary = df[df[EVAL_TYPE] == ARGUMENT]
    else:
        df_summary = df
    
    columns_all = df_summary.columns.tolist()

    # Define broader categories
    df_summary[GROUP] = df_summary.apply(get_groups, axis=1)
    df_summary[CATEGORY] = df_summary.apply(get_category, axis=1)

    # Aggregate by category
    agg_dict = {NT: 'sum', NP: 'sum', TP: 'sum', FP: 'sum', FN: 'sum'}
    
    for c in columns_all:
        if (c not in agg_dict) and (c not in CATEGORY):
            agg_dict[c] = 'first'
            
    df_summary = df_summary.groupby([CATEGORY], as_index=False).agg(agg_dict)
    
    # Calculate metrics
    df_summary = df_f1(df_summary)
    
    '''
    Save results
    '''
    fn = os.path.join(path, SCORES_ALL)
    df.to_csv(fn, index=False)
    
    fn = os.path.join(path, SCORES_BY_EVENT)
    df_by_event.to_csv(fn, index=False)

    fn = os.path.join(path, SCORES_SUMMARY)
    df_summary.to_csv(fn, index=False)

    return (df, df_by_event, df_summary)

def get_event_counts(events):
    '''
    
    
    Parameters
    ----------
    events: list of list of Event (list of sentence of event)
    '''    
    assert isinstance(events, list)
    if len(events) > 0:
        assert isinstance(events[0], list)
        flat = [event for sent in events for event in sent]
        if len(flat) > 0:
            assert isinstance(flat[0], Event)
    
    # Get event types
    event_types = [event.type_ for sent in events for event in sent]
    event_types = sorted(list(set(event_types)))
    
    # Number of sentences
    n_sent = len(events)
    
    # Initialize event counts
    counts = OrderedDict()
    for evt_typ in event_types:
        counts[evt_typ] = [0]*n_sent

    # Iterate over sentences   
    for i_sent, sent in enumerate(events):
        for event in sent:
            trig = event.trigger()            
            if trig is not None:
                counts[trig.label][i_sent] += 1

    return counts



def get_event_data(corpus, norm, event_args, data_spec, path, name, \
                max_len=None, by_doc=True):
    '''
    Get data from corpus
    '''
    
    if len(data_spec) == 0:
        return ([], [], [])
    
    # Input text
    X = corpus.sents( \
                    norm = norm, 
                    max_len = max_len,
                    by_doc = by_doc,
                    data_spec = data_spec)

    # Document-level labels
    y = corpus.events( \
                    event_args = event_args,
                    max_len = max_len,
                    by_doc = by_doc,
                    data_spec = data_spec)

    # Document IDs
    i = corpus.doc_ids( \
                    data_spec = data_spec)


    assert len(X) == len(y), '{} vs {}'.format(len(X), len(y))
    assert len(X) == len(i), '{} vs {}'.format(len(X), len(i))

    # Get document and label summaries
    samples = []
    label_sent = Counter()
    label_occur = Counter()
    label_token = Counter()
    events = []
    for X_, y_, i_ in zip(X, y, i):

        # Sample summary
        samp = OrderedDict()
        
        samp['id_'] = i_
                
        labs = Counter()
        events.append('')
        events.append(str(i_))
        for sent in y_:
            for evt in sent:
                labs[evt.type_] += 1
                events.append(str(evt))
       
        for lab, cnt in labs.items():            
            samp[lab] = cnt
        text = ' <SEP> '.join([' '.join(sent) for sent in X_])
        samp['text_'] = text
        
        samples.append(samp)
        
        # Label distribution, including token count
        for sent in y_:
            by_sent = Counter()
            for evt in sent:
                for arg in evt.arguments:
                    k = (evt.type_, arg.type_, arg.label)
                    by_sent[k] += 1
                    label_occur[k] += 1
                    label_token[k] += len(arg.tokens)
            for k in by_sent:
                label_sent[k] += 1

    fn = os.path.join(path, 'labeled_samples_{}.csv'.format(name))
    df = pd.DataFrame(samples)
    df.to_csv(fn, index=False)
    
    
    label_counts = []
    for k in label_occur:
        evt_typ, arg_typ, arg_lab = k
        label_counts.append((evt_typ, arg_typ, arg_lab, label_sent[k], label_occur[k], label_token[k]))
    cols = [EVENT_TYPE, ARGUMENT, LABEL, 'sentence count', 'occurrence count', 'token count']
    df = pd.DataFrame(label_counts, columns=cols)
    fn = os.path.join(path, 'label_dist_{}.csv'.format(name))
    df = df.fillna(NONE_PH)
    df.to_csv(fn, index=False)

    
    events = "\n".join(events)
    fn = os.path.join(path, 'events_{}.txt'.format(name))
    with open(fn,'w') as f:
        f.write(events)


    
    return (X, y, i)    
    
