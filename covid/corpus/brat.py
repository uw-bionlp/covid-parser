
import re
import glob
import os
import numpy as np
import logging
from collections import Counter
from collections import OrderedDict
from pathlib import Path
import stat
import copy
import re

from constants import *

from corpus.tokenization import tokenize_doc

COMMENT_RE = re.compile(r'^#')
TEXTBOUND_RE = re.compile(r'^T\d+')
EVENT_RE = re.compile(r'^E\d+\t')
ATTRIBUTE_RE = re.compile(r'^A\d+\t')
HAS_ATTR = [STATUS]

class Attribute(object):
    '''
    Container for attribute
    
    annotation file examples:
        A1      Value T2 current
        A2      Value T8 current
        A3      Value T9 none
        A4      Value T13 current
        A5      Value T17 current
    '''
    def __init__(self, id_, attr, textbound, value):
        self.id_ = id_
        self.attr = attr
        self.textbound = textbound
        self.value = value
    
    def __str__(self):
        return str(self.__dict__)
    
    def __eq__(self, other):
        return (self.attr == other.attr) and \
               (self.textbound == other.textbound) and \
               (self.value == other.value)

class Textbound(object):
    '''
    Container for textbound
    
    Annotation file examples:
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
    '''
    def __init__(self, id_, type_, start, end, text):
        self.id_ = id_
        self.type_ = type_
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        return str(self.__dict__)


    def token_indices(self, char_indices):
        i_sent, (out_start, out_stop) = find_span(char_indices, self.start, self.end)
        return (i_sent, (out_start, out_stop))


class Event(object):
    '''
    Container for event
    
    Annotation file examples:
        E3      Family:T7 Amount:T8 Type:T9
        E4      Tobacco:T11 State:T10
        E5      Alcohol:T13 State:T10
        E6      Drug:T14 State:T10
        E1      Tobacco:T2 State:T1
        E2      Alcohol:T5 State:T4
        
        id_     event:head (entities)
    '''
    
    def __init__(self, id_, type_, arguments):
        self.id_ = id_
        self.type_ = type_
        self.arguments = arguments
    
    def __str__(self):
        return str(self.__dict__)


#def get_attributes(attr_dict, tb_id, allowable_attr, arg_type):
def get_attributes(attr_dict, tb_id, arg_type):
    '''
    
    
    
    Parameters
    ----------
    attr_dict: dictionary of Attribute
    tb_id: text bound ID as string
    allowable_attr: dictionary of attribute assignments by argument type
    arg_type: argument type as string
    
    
    '''
    
    # Get applicable attributes for textbound and argument type
    attributes = []
    for id_, attr in attr_dict.items(): 
        #if (attr.textbound == tb_id) and \
        #   (arg_type in allowable_attr) and (allowable_attr[arg_type] == attr.attr):
        if attr.textbound == tb_id:
            attributes.append(attr)
    
    # Number of attributes per text bound
    attr_count = len(attributes)

    # If no attributes, return dummy attribute        
    if attr_count == 0:
        return Attribute(None, None, None, None)

    
    #elif (attr_count == 1) and (False):
    #    return attributes[0]
    
    else:
        n = len(arg_type)
        for attr_obj in attributes:
            #if arg_type in attr_obj.attr:
            if arg_type == attr_obj.attr[:n]:
                return attr_obj

        #msg = []
        #msg.append("multiple diff attributes found:\ttb_id={}, arg_type={}\n{}".format(tb_id, arg_type, '\n'.join([str(a) for a in attributes])))
        #msg.append("arg_type:\t{}".format(arg_type))                              
        #msg = "\n".join(msg)
        #raise ValueError(msg)
        #logging.warn(msg)
        return Attribute(None, None, None, None)

def get_annotations(ann):
    '''
    Load annotations, including taxbounds, attributes, and events
    
    ann is a string
    '''
    
    # Parse string into nonblank lines
    lines = [l for l in ann.split('\n') if len(l) > 0]
       
    
    # Confirm all lines consumed
    remaining = [l for l in lines if not \
            ( \
                COMMENT_RE.search(l) or \
                TEXTBOUND_RE.search(l) or \
                EVENT_RE.search(l) or \
                ATTRIBUTE_RE.search(l)
            )
        ]
    msg = 'Could not match all annotation lines: {}'.format(remaining)
    assert len(remaining)==0, msg

    # Get events 
    events = parse_events(lines)
    
    # Get text bounds    
    textbounds = parse_textbounds(lines)
    
    # Get attributes
    attributes = parse_attributes(lines)

    return (events, textbounds, attributes)
    
def parse_textbounds(lines):
    """
    Parse textbound annotations in input, returning a list of
    Textbound.

    ex. 
        T1	Status 21 29	does not
        T1	Status 27 30	non
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
        T7	Type 78 90	recreational
        T8	Drug 91 99	drug use

    """

    textbounds = {}
    for l in lines:
        if TEXTBOUND_RE.search(l):
            
            # Split line 
            id_, type_start_end, text = l.split('\t')

            # Check to see if text bound spans multiple sentences
            mult_sent = len(type_start_end.split(';')) > 1
            
            # Multiple sentence span, only use portion from first sentence
            if mult_sent:
                type_start_end = type_start_end.split(';')[0]
                logging.warn('')
                logging.warn('Parse_textbounds - tb spanning multiple sentences. only using tb from first sentence')
                logging.warn('\t{}'.format(l))
            
            # Split type and offsets
            type_, start, end = type_start_end.split()

            # Convert start and stop indices to integer
            start, end = int(start), int(end)
            
            # Multiple sentence span, only use portion from first sentence
            if mult_sent:
                text_orig = text
                text = text_orig[0:end-start]
                logging.warn('Only using "{}" from "{}"'.format(text, text_orig))                

            # Build text bound object            
            textbounds[id_] = Textbound(
                          id_ = id_,
                          type_= type_, 
                          start = start, 
                          end = end, 
                          text = text,
                          )
    
    return textbounds

def parse_attributes(lines):
    """
    Parse attributes, returning a list of Textbound.
        Assume all attributes are 'Value' 
        
        ex.

        A2      Value T4 current
        A3      Value T9 past
        A4      Value T13 none
        A1      Value T2 current
        A2      Value T8 current
        A3      Value T9 none
        A4      Value T13 current
        A5      Value T17 current
        A1      Value T4 current
        A2      Value T10 current
        A3      Value T11 none

    """

    attributes = {}
    for l in lines:

        if ATTRIBUTE_RE.search(l):
            
            # Split on tabs
            id_, attr_textbound_value = l.split('\t')
            
            attr, textbound, value = attr_textbound_value.split()
            
            # Add attribute to dictionary
            attributes[id_] = Attribute( \
                    id_ = id_,
                    attr=attr, 
                    textbound=textbound, 
                    value=value)
    return attributes


def parse_events(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
        E2      Tobacco:T7 State:T6 Amount:T8 Type:T9 ExposureHistory:T18 QuitHistory:T10
        E1      Tobacco:T2 State:T1
        E2      Alcohol:T5 Amount:T6 State:T4
        E3      LivingSituation:T7 Method:T9 State:T10
        E4      Residence:T11 State:T10 Location:T8
        E5      Family:T12 Type:T13
        E1      Occupation:T1 State:T2 Method:T3
        E2      MaritalStatus:T4 State:T5 Type:T6
        E3      Family:T7 Amount:T8 Type:T9
        E4      Tobacco:T11 State:T10
        E5      Alcohol:T13 State:T10
        E6      Drug:T14 State:T10
        E1      Tobacco:T2 State:T1
        E2      Alcohol:T5 State:T4
        E7      EnvironmentalExposure:T19 Type:T20 Amount:T21 State:T22
        E4      Tobacco:T12 State:T11
        E5      Alcohol:T15 State:T14
        E6      SexualHistory:T18 Type:T19 Time:T17 State:T16
        E1      Family:T1 Amount:T2 Type:T3
        E2      MaritalStatus:T4 State:T6 Type:T5
        E1      Tobacco:T2 State:T1
        E1      Tobacco:T2 State:T1
        E2      Alcohol:T5 State:T4
        E3      Drug:T7 State:T6
        E1      LivingSituation:T1 Method:T5 State:T2
        E2      Family:T3 Type:T4
        E3      MaritalStatus:T6 Type:T7 State:T8
        E4      Occupation:T9 State:T12 Location:T10 Type:T11
    
        id_     event:tb_id ROLE:TYPE ROLE:TYPE ROLE:TYPE ROLE:TYPE
    """

    events = {}
    for l in lines:
        if EVENT_RE.search(l):

            # Split based on white space            
            entries = [tuple(x.split(':')) for x in l.split()]
            
            
            # Get ID
            id_ = entries.pop(0)[0]

            # Entity type
            event_type, _ = tuple(entries[0])
                        
            # Role-type
            arguments = OrderedDict()
            for i, (argument, tb) in enumerate(entries):
                
                # Assume first argument is that trigger
                if i == 0:
                    arg_name = TRIGGER
                
                # Map state to status
                #elif argument == STATE:
                #    arg_name = STATUS
                
                # Assume all other arguments are entities (multiword spans)
                else:
                    arg_name = argument
                    #pass #arg_name = ENTITY

                # Remove trailing integers (e.g. Status2)
                #arg_name = arg_name.rstrip('123456789')
    
                arguments[arg_name] = tb


            # Only include desired arguments
            events[id_] = Event( \
                      id_ = id_,
                      type_ = event_type,
                      arguments = arguments)
            
    return events



def annotation_counts(events, textbounds, \
    occur_counts=None, word_counts=None):
    '''
    Get annotation counts (occurrence and word counts)
    '''

    if occur_counts is None:
        occur_counts = {}
    if word_counts is None:
        word_counts = {}
    
    # Loop on events       
    for id_, evt_rep in events.items():

        # Get event
        event = evt_rep.type_
        entities = evt_rep.arguments
   
        # Loop on entities associated with event
        for entity, tb in entities.items():

            # Initialize counter
            if (event, entity) not in word_counts.keys():
                occur_counts[(event, entity)] = 0
                word_counts[(event, entity)] = 0   

            # Increment occurrence counts
            occur_counts[(event, entity)] += 1
            
            # Text associated with text bound
            text = textbounds[tb].text

            # Increment word counts
            wc = sum([len(sent) for sent in tokenize_doc(text)])
            word_counts[(event, entity)] += wc
    
    return (occur_counts, word_counts)        


def merged_status(evt_dict, tb_dict, indices):
    '''
    Get annotation counts (occurrence and word counts)
    '''

    # Sentence indices by event
    sent_idxs = {}
    
    # Loop on events       
    for id_, evt_rep in evt_dict.items():

        # Get event
        event = evt_rep.type_
        entities = evt_rep.arguments

        # Add event to count dictionary
        if event not in sent_idxs.keys():
            sent_idxs[event] = []
        
        # Process if status value present
        if STATUS in entities.keys():
            tb = tb_dict[entities[STATUS]]

            # Find index of sentence associated with status
            idx = -1
            for i, sent in enumerate(indices):
                for tok_start, tok_end in sent:
                    if (tb.start >= tok_start) and \
                       (tb.start < tok_end) and \
                       (idx == -1):
                        idx = i
                        break
            if idx != -1:                                           
                sent_idxs[event].append(idx)
                

    merged_counts = {}
    for event, indices in sent_idxs.items():
        merged_counts[event] = len(indices) - len(set(indices))
    
    return merged_counts


def get_files(path, ext='.', relative=False):
    files = list(Path(path).glob('**/*.{}'.format(ext)))
    
    if relative:
        files = [os.path.relpath(f, path) for f in files]
    
    return files

def get_brat_files(path):
    '''
    Find text and annotation files
    '''
    # Text and annotation files
    text_files = get_files(path, TEXT_FILE_EXT, relative=False)
    ann_files = get_files(path, ANN_FILE_EXT, relative=False)
           
    # Check number of text and annotation files
    msg = 'Number of text and annotation files do not match'
    assert len(text_files) == len(ann_files), msg

    # Sort files
    text_files.sort()    
    ann_files.sort()

    # Check the text and annotation filenames
    mismatches = [str((t, a)) for t, a in zip(text_files, ann_files) \
                                           if not filename_check(t, a)]
    fn_check = len(mismatches) == 0
    assert fn_check, '''txt and ann filenames do not match:\n{}'''. \
                        format("\n".join(mismatches))

    return (text_files, ann_files)


def get_filename(path):
    root, ext = os.path.splitext(path)
    return root

def filename_check(fn1, fn2):
    '''
    Confirm filenames, regardless of directory or extension, match
    '''
    fn1 = get_filename(fn1)
    fn2 = get_filename(fn2)
    
    return fn1==fn2    

def intersect(x_start, x_end, y_start, y_end):
    '''
    Look for intersection between x and y ranges
    '''
    # Look for intersection
    x_idx = range(x_start, x_end)
    y_idx = range(y_start, y_end)
    return len(set(x_idx) & set(y_idx)) > 0

def closest_index(vals, target):

    if target in vals:
        return target
    
    closest =  min(vals, key=lambda x:abs(x-target))
    
    logging.warn("No exact match: target={}, closest={}".format( \
                                          target, closest))
    return closest


def less_or_equal(vals, target):

    closest = None
    for v in vals:
        if v <= target:
            closest = v       
    
    return closest
                
def greater_or_equal(vals, target):

    closest = None
    for v in vals[::-1]:
        if v >= target:
            closest = v       

    return closest


def find_span(doc_idx, start, end):
    '''
    
        seq_of_idx = list of tuple of character indices of tokens
    '''

    # Not in a span
    in_span = False
    
    # Find closest match
    start_idx, end_idx = zip(*[(st, en) for sent in doc_idx \
                                                 for (st, en) in sent])
    target_char_start = less_or_equal(start_idx, start)
    target_char_end =  greater_or_equal(end_idx, end)

    # Loop on sentences in document
    for i_sent, sent_idx in enumerate(doc_idx):
        
        # Loop on tokens in sentence
        for i_tok, (tok_char_start, tok_char_end) in enumerate(sent_idx):
        
            # Start char match        
            if (target_char_start == tok_char_start):
                
                # Make sure not already in span
                assert not in_span, "already in span?"
                
                # Token index of span start
                out_start = i_tok
             
                # In a span
                in_span = True
                 
            # 
            end_match = (target_char_end == tok_char_end)
            reached_end = (i_tok == len(sent_idx) - 1) and (in_span)
            
            # End char match
            if end_match:
                assert in_span == True    
                                             
            # Reached end of sentence, but still in span    
            elif reached_end and not end_match:
                logging.warn('Span crosses sentence boundary. Truncating span')        
            
            # Found match or reached end, return result
            if end_match or reached_end:
                    
                # Token index of span end
                out_stop = i_tok + 1
                
                assert out_stop > out_start
                
                # Exit with result
                return (i_sent, (out_start, out_stop))
    
    # Match sure span found
    assert False, "could not find span"
    
    return None    


def path_to_info(path):

    split_path = path.split(os.sep)
    id_ = split_path.pop(-1)
    annotator = split_path.pop(-1)
    round_ = split_path.pop(-1)    
    
    return (round_, annotator, id_)


#def mapper(x, map_=None):

#    # Use original span type
#    if (map_ is not None) and (x in map_):
#        x = map_[x]
#
#    return x
        

def mapper(evt_type, span_type, map_):
    
    if (map_ is not None) and \
       (evt_type in map_) and \
       (span_type in map_[evt_type]):
        span_type = map_[evt_type][span_type]
    
    return span_type
          
 
                        
def textbound_str(id_, type_, start, end, text):
    '''
    Create textbounds during from span
    
    Parameters
    ----------
    id_: current textbound id as string
    span: Span object
    
    Returns
    -------
    BRAT representation of text bound as string
    '''

    
    return 'T{id_}\t{type_} {start} {end}\t{text}'.format( \
        id_ = id_, 
        type_ = type_, 
        start = start, 
        end = end, 
        text = text)


def attr_str(attr_id, arg_type, tb_id, value):
    '''
    Create attribute string
    '''
    return 'A{attr_id}\t{arg_type} T{tb_id} {value}'.format( \
        attr_id = attr_id, 
        arg_type = arg_type,
        tb_id = tb_id, 
        value = value)
        
        
def event_str(id_, event_type, textbounds):
    '''
    Create event string
    
    Parameters:
    -----------
    id_: current event ID as string
    event_type: event type as string
    textbounds: list of tuple, [(span.type_, id), ...]

    '''
    
    # Start event string
    out = 'E{}\t'.format(id_)
    
    # Create event representation
    event_args = []
    for i, (arg_type, tb_id) in enumerate(textbounds):
        out += '{}:T{} '.format(arg_type, tb_id)
        
    return out

def file_permissions(path):
    
    os.chmod(path, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH | \
                   stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def batch_folder(fn, per_batch=1000):
    return '{0:05d}'.format(int(int(float(fn)/per_batch)*per_batch))


def brat_folder(path, id_, use_batch=False, idx=None, per_batch=1000):


    if use_batch:
        if idx is None:
            idx = id_
        
        d = batch_folder(fn=idx, per_batch=per_batch)
        path = os.path.join(path, d)


    if not os.path.exists(path):
        os.makedirs(path)

    return path

def write_file(path, id_, content, ext):
    
    # Output file name
    fn = os.path.join(path, '{}.{}'.format(id_, ext))
    
    # Directory, including path in id_
    dir_ = os.path.dirname(fn)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    # Write file        
    with open(fn, 'w', encoding=ENCODING) as f:
        f.write(content)       


    #os.chmod(dir_, stat.S_IWGRP)
    #os.chmod(fn, stat.S_IWGRP)

    
def write_txt(path, id_, text):
    '''
    Write text file
    '''        
    write_file(path, id_, text, TEXT_FILE_EXT)
        

def write_ann(path, id_, ann):
    '''
    Write annotation file
    '''    
    write_file(path, id_, ann, ANN_FILE_EXT)

def create_ann_str(events, arg_map=None, tb_map=None, attr_map=None,
            add_missing_trig=True):

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
    iE = 1
    iA = 1
    iT = 1
    
    # Initialize list of invitation strings
    output = []
    
    # Loop on sentences
    for sent in events:
        
        # Loop on events in sent
        for event in sent:

            # Loop on         
            args = []
            attrs = []

            # Current event type
            evt_type = event.type_                        

            # Add missing trigger if not present
            if add_missing_trig:
                trig_spans = [span for span in event.arguments if span.type_ == TRIGGER]
                missing_trig = len(trig_spans) == 0
                if missing_trig:
                    warning = []
                    warning.append('Event missing trigger. Adding trigger as first argument. \n')
                    warning.append('Original event: \n{}'.format(event))
                    trig_span = copy.deepcopy(event.arguments[0])
                    trig_span.type_ = TRIGGER
                    trig_span.label = evt_type                    
                    event.arguments.insert(0, trig_span)
                    warning.append('Updated event: \n{}'.format(event))
                    warning.append('\n')
                    logging.warn("\n".join(warning))
                        
            # Loop on spans for current event
            for i, span in enumerate(event.arguments):

                # Is trigger?
                is_trig = span.type_ == TRIGGER
              
                # Span type
                span_type = event.type_ if is_trig else span.type_
        
                # Text bound
                tb_type = mapper(evt_type, span_type, tb_map)
                tb = textbound_str( \
                        id_ = iT, 
                        type_ = tb_type,                                     
                        start = span.char_idxs[0], 
                        end = span.char_idxs[1], 
                        text = span.text)
                output.append(tb)
                
                # Argument for event
                arg_type = mapper(evt_type, span_type, arg_map)
                args.append((arg_type, iT))

                # Attribute
                if (not is_trig) and (span.label is not None):
                    attr_type = mapper(evt_type, span_type, attr_map)
                    attr = attr_str( \
                                    attr_id = iA, 
                                    arg_type = attr_type, 
                                    tb_id = iT, 
                                    value = span.label)
                    output.append(attr)
                    iA += 1

                iT += 1
                                         
            # Event
            if event.has_trigger() and (len(args) > 0):
                evt = event_str(iE, evt_type, args)
                output.append(evt) 
                iE += 1

    # Convert to string           
    output = "\n".join(output)
    
    return output

  