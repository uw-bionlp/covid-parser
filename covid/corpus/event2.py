import re
from collections import OrderedDict, Counter
import logging
import pandas as pd
from math import floor, ceil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from corpus.event import PRF, trig_first
from constants import *
from utils.scoring import add_params_to_df

PH = '--'

def flatten(X):
    return [x_ for x in X for x_ in x]

def count_events(events):

    # Loop on sentences
    counter = Counter()
    for sent in events:
        
        # Loop on events in current sentence
        for evt in sent:
            
            # Loop on arguments in current event
            for arg in evt.arguments:

                # Trigger span 
                if arg.type_ == TRIGGER:
                    s = 1
                
                # Labeled argument
                elif arg.label is not None:
                    s = 1
                
                # Span-only argument (token level scoring)
                else:
                    s = arg.tok_idxs[1] - arg.tok_idxs[0]
                
                # Update argument label, if None
                arg_lab = PH if arg.label is None else arg.label

                # Define counter key
                k = (evt.type_, arg.type_, arg_lab)

                # Increment counter
                counter[k] += s
                
    return counter        




def trig_dict(events):
    '''
    Convert from list of events to dictionary of events, 
    where trigger token indices are the key
    '''

    # Loop on events in sentence
    out = OrderedDict()    
    for evt in events:
        
        # Get key
        k = evt.trigger().tok_idxs
        
        # Issue warning, if key present
        if k in out:
            logging.warn("tok idx {} in {}".format(k, out.keys()))    
            #for evt in events:
            #    logging.warn("\t\n{}".format(evt))    
        
        # Add event        
        out[k] = evt

    return out

def overlap(a, b):
    
    a = set(list(range(*a)))
    b = set(list(range(*b)))
    ol = a.intersection(b)
    
    return len(ol)

def count_matches(eventsA, eventsB):
    
    # Loop on sentences
    counter = Counter()
    for A, B in zip(eventsA, eventsB):
        
        # Convert from list of events to dictionary of events, 
        # where trigger token indices are the key
        A = trig_dict(A)
        B = trig_dict(B)
    
        # Get overlapping token indices
        intersect = list(set(A.keys()).intersection(set(B.keys())))
    
        # Loop on trigger token indices present in both A and B
        for tok_idxs in intersect:
            
            # Get arguments associated with specific event
            evtA = A[tok_idxs]
            evtB = B[tok_idxs]

            # Event types match
            match_evt = evtA.type_ == evtB.type_
            
            # Get arguments associated with specific event           
            argsA = evtA.arguments[:]
            argsB = evtB.arguments[:]

            # Loop on arguments in event A
            for a in argsA:
                
                # Loop on remaining arguments in event B
                for i, b in enumerate(argsB):
                    
                    
                    match_arg = a.type_ == b.type_
                    match_lab = a.label == b.label
                    
                    match_evt_arg_lab = match_evt and match_arg and match_lab
                    ol = overlap(a.tok_idxs, b.tok_idxs)
                    match_tok     = a.tok_idxs == b.tok_idxs
                    match_sent    = a.sent_idx == b.sent_idx
                    
                    match = False
                    s = 0
                    
                    # Trigger scoring
                    if match_evt_arg_lab and (a.type_ == TRIGGER):
                        s = 1
                        match = True
                        
                    # Labeled argument scoring
                    elif match_evt_arg_lab and (a.label is not None):
                        s = 1
                        match = True
                    
                    # Span-only argument scoring
                    elif match_evt_arg_lab and match_sent and (a.label is None) and (ol > 0):
                        s = ol
                        match = True
                    
                    # No match
                    else:
                        pass                        
            
                    # Update counter
                    if match:
                        a_lab = PH if a.label is None else a.label
                        
                        counter[(evtA.type_, a.type_, a_lab)] += s
                        argsB.pop(i)
                        break
    return counter

def score_events(gold, pred, by_doc=False):
    
    assert len(gold) == len(pred)
    
    if by_doc:
        gold = flatten(gold)
        pred = flatten(pred)

    assert len(gold) == len(pred)        
        
    nt = count_events(gold)
    np = count_events(pred)
        
    tp = count_matches(gold, pred)
    
    K = set(nt.keys()).union(set(np.keys())).union(set(tp.keys()))
    K = sorted(list(K))
    
    counts = []
    for k in K:
        counts.append([ARGUMENT] + list(k) + [nt[k], np[k], tp[k]])

    columns = [EVAL_TYPE, EVENT_TYPE, ARGUMENT, LABEL, NT, NP, TP]
    df = pd.DataFrame(counts, columns=columns)        
    
    df[FN] = df[NT] - df[TP]
    df[FP] = df[NP] - df[TP]
    df[P] = df[TP].astype(float)/(df[NP].astype(float))
    df[R] = df[TP].astype(float)/(df[NT].astype(float))
    df[F1] = 2*df[P]*df[R]/(df[P] + df[R])
    
    return df

def len_check(x, y):
    assert len(x) == len(y), "length mismatch: {} vs {}".format(len(x), len(y))    


class EventScorer(object):
    
    def __init__(self, 
                by_doc = True,             
                ):
        '''
        
        
        Parameters
        ----------

        true_spans_sep: Boolean indicating whether events and spans 
                        are provided in 'true' in fit
        pred_spans_sep: Boolean indicating whether events and spans 
                        are provided in 'pred' in fit            
        by_doc: indicates whether 'true' and 'pred' in fit are provided 
                        at document level, e.g. [docs [sentences [events]]] 
                        if false, assumed to be of the form [sentences [events]]
        
        '''     
        self.by_doc = by_doc
        
    def fit(self, true, pred, params=None):
        '''        
        Score predictions
        '''

        len_check(true, pred)
        
        # If provided as list of documents, flatten, so list of sentences
        flatten = lambda docs: [sent for doc in docs for sent in doc]    

        # Flatten events
        if self.by_doc:
            true = flatten(true)
            pred = flatten(pred)

        # Check sentence count
        len_check(true, pred)

        # Get counts
        df = score_events2( \
                    gold = true, 
                    pred = pred,
                    by_doc = False)
        
        # Include each parameter in data frame       
        if params is not None:
            df = add_params_to_df(df, params)        

        return df
            


def perf_plot(path, df, name,

    plot_groups = None,    
    figsize=(3.0, 2.3),
    font_family = 'serif',
    font_type = 'Times New Roman',    
    font_size = 7, 
    font_size_legend = 7,
    barWidth = 0.3,
    bar_fract = 0.85,
    left=0.18, 
    bottom=0.45, 
    right=0.98, 
    top=0.97,
    wspace = 0.1,
    xlab_coord = -0.76,
    sep_ub = -0.05,
    sep_lb = -0.7,
    sep_wd = 0.5,
    sep_color = 'grey',
    rotation = 90,
    ha = 'center',
    bbox_to_anchor=(1.02, 0.5),
    map_ = {},
    pad = 0.1,
    labelspacing = 0.0,
    borderpad = 0.1,
    framealpha=1.0,
    color = sns.color_palette("colorblind", n_colors=8),
    edgecolor = None, 
    metric = F1, 
    ytick_min = 0.0,
    ytick_max = 1.0,
    ytick_spacing = 0.1,
    ):
    
    # Font configuration
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.serif'] = [font_type] + plt.rcParams['font.serif']
   
    # Agg across labels

    # Argument-level only
    if EVAL_TYPE in df:
        df = df[df[EVAL_TYPE]==ARGUMENT]

    
    gb = [EVENT_TYPE, ARGUMENT, ROUND]
    df = df.groupby(gb).sum().reset_index()
    df = PRF(df)

   
    
    if plot_groups:
        
        df['group'] = df[ARGUMENT].map(plot_groups)
        df = df.sort_values([EVENT_TYPE, 'group', ARGUMENT])
    else:        
        # Sort argument, so trigger first, then alphabetical    
        df = trig_first(df)
        df['group'] = 0
    
    # All event types
    event_types = sorted(df[EVENT_TYPE].unique().tolist())

    # Rounds
    rounds = sorted(df[ROUND].unique().tolist())

    # Number of columns per event type
    width_ratios = [len(df[df[EVENT_TYPE] == evt_typ]) for evt_typ in event_types]
    
    # Width ratios for subplots
    gridspec_kw={'width_ratios': width_ratios}

    # Creat figure
    fig, axs = plt.subplots(1, len(event_types), \
                        figsize = figsize,  
                        sharey = 'all', 
                        gridspec_kw = gridspec_kw)
    plt.subplots_adjust( \
                        left = left, 
                        bottom = bottom, 
                        right = right, 
                        top = top, 
                        wspace = wspace)    

    # Loop on event types    
    rects = []
    for i, evt_typ in enumerate(event_types):
        
        # Current axis
        ax = axs[i]
        
        # Event type subset
        df_evt = df[df[EVENT_TYPE] == evt_typ]
        
        # Loop on rounds
        for j, rnd in enumerate(rounds):
        
            # Round subset
            df_rnd = df_evt[df_evt[ROUND] == rnd].reset_index()

            # Get XY coordinates            
            x = np.arange(len(df_rnd)) + j*barWidth
            y = df_rnd[metric]
            
            # Add bar
            rect = ax.bar(x, y, color=color[j], width=barWidth*bar_fract, edgecolor=None, label=rnd)

            # Create X labels
            if j == 0:
                xlabels = [map_.get(a, a) for a in df_rnd[ARGUMENT]]
                sep_marks = df_rnd['group'].diff()[lambda x: x != 0].index.tolist()[1:]
                
            
            # Get rectangle for legend
            if (i == 0) and (j == 0):
                rects.append(rect)
                
        # Define X tick marks
        xticks = np.arange(len(df_rnd)) + (len(rounds)-1)/2*barWidth
        ax.set_xticks(xticks)

        yticks = np.arange(ytick_min, ytick_max+ytick_spacing, ytick_spacing)
        ax.set_yticks(yticks)
        
        for m in sep_marks:
            t = xticks[m] - 0.5
            line = plt.Line2D([t, t], [sep_ub, sep_lb], color=sep_color, linewidth=sep_wd)
                      #transform=ax.get_xaxis_transform(), color='black', linewidth=linewidth) 
            line.set_clip_on(False)
            ax.add_line(line)

        # Define X tick labels        
        ax.set_xticklabels(xlabels, rotation=rotation, ha=ha)

        # Define X label (event type)
        ax.set_xlabel(map_.get(evt_typ, evt_typ), fontweight='bold', va='top')
        ax.xaxis.set_label_coords(0.5, xlab_coord)

        # Adjust Y limits
        a, b = ax.get_ylim()
        a = floor(a*10)/10
        b = ceil(b*10)/10
        ax.set_ylim(a, b)

        # Add xticks on the middle of the group bars
        if i == 0:
            ax.set_ylabel(metric, fontweight='bold')
        ax.yaxis.grid(color='gray', linewidth=0.5)

        # Move axis to back
        ax.set_axisbelow(True)

        # Create legend
        if (i == len(event_types) - 1) and (j == len(rounds) - 1):
            plt.legend(handles=rects)
            ax.legend(loc='upper right', \
                        handletextpad=pad, 
                        labelspacing=labelspacing, 
                        borderpad=borderpad, 
                        framealpha=framealpha,
                        prop={'size': font_size_legend})


    fn = os.path.join(path, 'perf_{}.csv'.format(name))
    df.to_csv(fn)
    
    fig = ax.get_figure()
    fn = os.path.join(path, 'perf_{}.pdf'.format(name))
    fig.savefig(fn, quality=100)
    fn = os.path.join(path, 'perf_{}.png'.format(name))
    fig.savefig(fn, quality=100, dpi=800)




def event_trig_filter_doc(events, trig_filt=None): 
    '''
    Filter events based on trigger span tokens
    
    
    Parameters
    ----------
    events: list of list of events (doc of sentences of events)
    trig_filt: dictionary of list of allowable trigger spans
                {'SSx': ['pain', 'headache', ...]}
    '''
        
    # Loop on sentences in current document
    new_doc = []
    for sent in events:

        # Add list for current sentence
        new_doc.append([])
        
        # Loop on events in current sentence
        for evt in sent:
            
            # Get trigger text
            trig_txt = " ".join(evt.trigger().tokens).lower()

            # Check trigger text
            if (trig_filt is None) or \
               (evt.type_ not in trig_filt) or \
               (trig_txt in trig_filt[evt.type_]):
                
                new_doc[-1].append(evt)                
        
    return new_doc




def event_trig_filter(events, trig_filt=None): 
    '''
    Filter events based on trigger span tokens
    
    
    Parameters
    ----------
    events: list of list of list of events (corpus of doc of sentences of events)
    trig_filt: dictionary of list of allowable trigger spans
                {'SSx': ['pain', 'headache', ...]}
    '''    

    # Loop on documents
    new_events = []
    for doc in events:
        new_doc = event_trig_filter_doc(doc, trig_filt=trig_filt)    
        new_events.append(new_doc)
        
    return new_events    


def trig_span_adj_doc(events, event_types=None):
    '''
    Set trigger spans to first token in current trigger span, 
    based on list of event types
    
    Parameters
    ----------
    events: list of list of events (doc of sentences of events)
    event_types: list of event types to for which to truncate trigger spans
    '''

    if event_types is None:
        return events

    else:

        # Loop on sentences in document
        for sent in events:
            

            # Loop on events in current sentence
            for evt in sent:
                
                # If trigger in list, adjust
                trig = evt.trigger()
                if trig.label in event_types:
                    trig.tokens = [trig.tokens[0]]
                    trig.tok_idxs = (trig.tok_idxs[0], trig.tok_idxs[0]+1)
        
        return events

def trig_span_adj(events, event_types=None):
    '''
    Set trigger spans to first token in current trigger span, 
    based on list of event types
    
    Parameters
    ----------
    events: list of list of list of events (corpus of doc of sentences of events)
    event_types: list of event types to for which to truncate trigger spans
    '''
    if event_types is None:
        return events

    else:
        
        for doc in events:
            trig_span_adj_doc(doc, event_types)
        return events
                



   
def swap_loc(events, trig_loc_swap):
    
    if trig_loc_swap is None:
        return events
    
    # Loop on documents
    for doc in events:
        
        # Loop on sentences in current doc
        for sent in doc:
            
            # Loop on events in current sentence
            for evt in sent:
                
                # Get trigger
                trig = evt.trigger()
                
                # Loop on arguments in current event
                for arg in evt.arguments:
                    
                    # Event/argument meets criteria for swapping
                    if (evt.type_ in trig_loc_swap) and \
                       (arg.type_ in trig_loc_swap[evt.type_]):
                        
                        #print()
                        #print(arg)
                        arg.sent_idx = trig.sent_idx
                        arg.tok_idxs = trig.tok_idxs
                        arg.tokens   = trig.tokens    
                        
                        #print(arg)

    return events                
    


def arg2tuple(trig, arg):
    '''
    Tuple representations of trigger-argument pair
    '''
    
    rep = []
    rep.append(trig.label)
    rep.append(arg.type_)
    if arg.label is None:
        rep.append('--')
    else:
        rep.append(arg.label)

    rep.append(trig.sent_idx)
    rep.extend(list(trig.tok_idxs))
    
    if arg.label is None:
        
        reps = []
        for tok_idx in range(arg.tok_idxs[0], arg.tok_idxs[1]):
            r = rep[:]
            r.append(arg.sent_idx)
            r.append(tok_idx)
            reps.append(tuple(r))

    else:
        reps = [tuple(rep)]

    return reps



def counts_from_tuples(tuples):
    
    c = Counter()
    for tup in tuples:
        t = tuple(list(tup)[0:3])
        c[t] += 1

    return c

def evt2tuples(event):
    '''
    Tuple representations of event
    '''
    
    trig = event.trigger()
    
    reps = []
    for arg in event.arguments:
        reps.extend(arg2tuple(trig, arg))
    
    return reps
    

def events2tuples(events):
    '''
    Tuple representations of sequence of events
    '''
    reps = []
    for evt in events:
        reps.extend(evt2tuples(evt))

    return reps
    


    
def score_events2(gold, pred, by_doc=False):
    
    assert len(gold) == len(pred)

    if by_doc:
        flat = lambda A: [a_ for a in A for a_ in a]
        gold = flat(gold)
        pred = flat(pred)
        
    assert len(gold) == len(pred)
    

    # Loop on sentences 
    gc = Counter()
    pc = Counter()
    ec = Counter()
    for ge, pe in zip(gold, pred):        
        
        # Gold events as tuple
        gt = events2tuples(ge)
        
        # Predicted events as tuple
        pt = events2tuples(pe)
        
        # Equivalent tuples
        et = list(set(gt).intersection(set(pt)))
        
        # Gold counts
        gc += counts_from_tuples(gt)
        
        # Predicted counts
        pc += counts_from_tuples(pt)
        
        # Equivalent counts
        ec += counts_from_tuples(et)
                
    # All keys (event type, argument type, label) combinations    
    all_keys = set(list(gc.keys()) + list(pc.keys()))
    
    agg = []
    for k in all_keys:
        evt_typ, arg_typ, arg_lab = k
        d = OrderedDict()
        d[EVENT_TYPE] = evt_typ
        d[ARGUMENT] = arg_typ
        d[LABEL] = arg_lab
        d[NT] = gc[k]
        d[NP] = pc[k]
        d[TP] = ec[k]
        
        agg.append(d)
    
    df = pd.DataFrame(agg)

    df[FN] = df[NT] - df[TP]
    df[FP] = df[NP] - df[TP]
    df[P] = df[TP].astype(float)/(df[NP].astype(float))
    df[R] = df[TP].astype(float)/(df[NT].astype(float))
    df[F1] = 2*df[P]*df[R]/(df[P] + df[R])
    
    df = df.sort_values([EVENT_TYPE, ARGUMENT, LABEL])
    
    
    return df
    
    