import re
import numpy as np

BIO_DEFAULT = True 


PAD_START = True 
PAD_END = True 



from constants import BEGIN, INSIDE, OUTSIDE, NEG_LABEL
from constants import START_TOKEN, END_TOKEN


def flatten_(Y):
    return [y for yseq in Y for y in yseq]

'''
Sequence padding
'''
# Pad start of sequence
def pad_start_seq(seq, pad):
    return [pad] + seq

# Pad start of sequences (document)
def pad_start_doc(sequences, pad):
    return [pad_start_seq(seq, pad) for seq in sequences]

# Pad end of sequence
def pad_end_seq(seq, pad):
    return seq + [pad]

# Pad and sequences (document)
def pad_end_doc(sequences, pad):
    return [pad_end_seq(seq, pad) for seq in sequences]

def strip_start_seq(seq):
    return seq[1:]

def strip_start_doc(sequences):
    return [strip_start_seq(seq) for seq in sequences]

def strip_end_seq(seq):
    return seq[0:-1]

def strip_end_doc(sequences):
    return [strip_end_seq(seq) for seq in sequences]


def add_BIO_seq(seq, begin, inside, outside):
    
    # Shift sequence to assess previous label
    seq_previous = [outside] + seq[0:-1]
    
    # Get new labels
    new_seq = []
    for label_previous, label_current in zip(seq_previous, seq):
        
        # Current label is negative
        if label_current == outside:
            new_seq.append(label_current)
        
        # Previous label not equal to current label, so add begin prefix
        elif label_current != label_previous: 
            new_seq.append(begin + label_current)
        
        # Previous label and current label equal
        elif label_current == label_previous:
            new_seq.append(inside + label_current)
        
        # Check for error
        else:
            raise ValueError("Error assigning BIO labels")

    return new_seq

def add_BIO_doc(sequences, begin, inside, outside):    
    
    return [add_BIO_seq(seq, begin, inside, outside) \
                                                   for seq in sequences]

def is_begin(x):
    return bool(re.match('^{}'.format(BEGIN), x))

def is_inside(x):
    return bool(re.match('^{}'.format(INSIDE), x))

def is_outside(x):
    return x == OUTSIDE
    
    

    

def strip_BIO_tok(x, begin=BEGIN, inside=INSIDE):

    return re.sub('(^{})|(^{})'.format(begin, inside), '', x)

def strip_BIO_seq(seq, begin=BEGIN, inside=INSIDE):

    return [strip_BIO_tok(label, begin=begin, inside=inside) for label in seq]



def strip_BIO_doc(sequences, begin=BEGIN, inside=INSIDE):
    return [strip_BIO_seq(seq, begin, inside) for seq in sequences]




def preprocess_labels_seq(seq, \
    BIO = BIO_DEFAULT, 
    begin = BEGIN, 
    inside = INSIDE,
    outside = OUTSIDE,
    pad_start = PAD_START,
    pad_end = PAD_END,
    ):
    
    if pad_start:
        seq = pad_start_seq(seq, outside)
    if pad_end:
        seq = pad_end_seq(seq, outside)    
    if BIO:
        seq = add_BIO_seq(seq, begin, inside, outside)    

    return seq


def preprocess_labels_doc(sequences, \
    BIO = BIO_DEFAULT, 
    begin = BEGIN, 
    inside = INSIDE,
    outside = OUTSIDE,
    pad_start = PAD_START,
    pad_end = PAD_END,
    ):

    return [preprocess_labels_seq(seq, BIO, begin, inside, outside, pad_start, pad_end) for seq in sequences]



def preprocess_tokens_seq(seq, \
    pad_start = PAD_START,
    pad_end = PAD_END,
    start_token = START_TOKEN,
    end_token = END_TOKEN,
    ):
    
    if pad_start:
        seq = pad_start_seq(seq, start_token)
    if pad_end:
        seq = pad_end_seq(seq, end_token)    
    return seq

def preprocess_tokens_doc(sequences, \
    pad_start = PAD_START,
    pad_end = PAD_END,
    start_token = START_TOKEN,
    end_token = END_TOKEN,
    ):
    
    
    return [preprocess_tokens_seq(seq, pad_start, pad_end, start_token, end_token) for seq in sequences]

def postprocess_tokens_seq(seq, \
    pad_start = PAD_START,
    pad_end = PAD_END,
    ):
    
    if pad_start:
        seq = strip_start_seq(seq)
    if pad_end:
        seq = strip_end_seq(seq)    
    return seq



def postprocess_tokens_doc(sequences, \
    pad_start = PAD_START,
    pad_end = PAD_END,
    ):
    

    return [postprocess_tokens_seq(seq, pad_start, pad_end) for seq in sequences]

def postprocess_labels_seq(seq, \
    BIO = BIO_DEFAULT, 
    begin = BEGIN, 
    inside = INSIDE,
    pad_start = PAD_START,
    pad_end = PAD_END,
    ):

    if pad_start:
        seq = strip_start_seq(seq)
    
    if pad_end:
        seq = strip_end_seq(seq)    
   
    if BIO:
        seq = strip_BIO_seq(seq, begin, inside)
    
        
    return seq


def postprocess_labels_doc(sequences, \
        BIO = BIO_DEFAULT, 
        begin = BEGIN, 
        inside = INSIDE,
        pad_start = PAD_START,
        pad_end = PAD_END,
        flatten = False,
    ):

    sequences = [postprocess_labels_seq(seq, BIO, begin, inside, pad_start, pad_end) for seq in sequences]

    if flatten:
        sequences = flatten_(sequences)
    
    return sequences


def BIO_prob_merge_tok(y, labels):

    x = {orig:0 for orig, _ in labels}
    for orig, BIO in labels:
        if BIO in y:
            x[orig] += y[BIO]
    return x

def BIO_prob_merge(Y, \
    begin = BEGIN, 
    inside = INSIDE,
    ):
    
    labels_BIO = list(set([k for y in Y for k in y.keys()]))
    labels_orig = strip_BIO_seq(labels_BIO, begin, inside)
    labels = list(zip(labels_orig, labels_BIO))
    
    X = []
    for y in Y:
        X.append(BIO_prob_merge_tok(y, labels))

    return X
    

def label_sent(labels, \
                hierarchy=None, 
                neg_label=OUTSIDE):
    '''
    Convert single sequence of labels (sentence-level) to a single label
    '''
    
    # Unique labels without negative label (i.e. None)
    unique = list(set(labels))
    
    # Length = 1, so return only label
    if len(unique) == 1:
        return unique[0]

    # Length = 2 and neg_label present, so return non-negative label
    elif (len(unique) == 2) and (neg_label in unique):
        unique.remove(neg_label)
        return unique[0]

    # Multiple labels found, so use hierarchy
    elif hierarchy is not None:
    
        # Indices of all values within hierarchy
        indices = [hierarchy.index(sf) for sf in unique]
        
        # Select status based on hierarchy
        found = hierarchy[min(indices)]

        # Issue warning
        msg = '\Status values selected:\n{}'.format(found)
        #warnings.warn(msg)
        
        return found
    
    else:
        print("lablels:", labels)
        raise ValueError("too many sentence labels:\t{}".format(unique))

def label_sentences(labels, \
                hierarchy=None, 
                neg_label=OUTSIDE):
    '''
    Convert a sequence of sequence labels (document-level) 
    to a sequence of labels
    '''
    return [label_sent(sent, hierarchy, neg_label) for sent in labels]