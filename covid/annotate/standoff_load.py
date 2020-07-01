
import sys
import os
import zipfile
import glob
import re
import numpy as np
from six import iteritems
import json
import joblib
import json
import copy
import warnings
from collections import Counter
import pandas as pd
import warnings




from nltk.tokenize import word_tokenize


from utils.tokenization_nltk import tokenize_doc, get_char_indices
from utils.misc import dict_to_list, list_to_dict
from utils.misc import nested_dict_to_list, list_to_nested_dict
from utils.seq_prep import preprocess_labels_doc, preprocess_tokens_doc
from utils.df_helper import filter_df
#from corpus.mimic import norm_deident


COMMENT_RE = re.compile(r'^#')
TEXTBOUND_RE = re.compile(r'^T\d+')
EVENT_RE = re.compile(r'^E\d+\t')
ATTRIBUTE_RE = re.compile(r'^A\d+\t')
TEXT_FILE_EXT = 'txt'
ANN_FILE_EXT = 'ann'
NEG_LABEL = 0

SUBSETS = ['train', 'tune', 'test']
SUBSET_DEFAULT = 'all'

ENTITY_MAP_LOAD = {'State': 'Status'}
ENTITY_MAP_CREATE = {v:k for k, v in ENTITY_MAP_LOAD.items()}




'''
class Tokens(object):
    
    #Container for text
    
    def __init__(self, sentences):
        
        self.sentences = sentences
    
    
    def tokens(self):
        return [[tok.token for tok in sent] for sent in self.sentences]
        
    def __str__(self):
        return "\n".join([" ".join(sent) for sent in self.tokens()])
'''
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
    

class Textbound(object):
    '''
    Container for textbound
    
    Annotation file examples:
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
    '''
    def __init__(self, id_, entity, start, end, text):
        self.id_ = id_
        self.entity = entity            
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        return str(self.__dict__)


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
    
    def __init__(self, id_, event, entities):
        self.id_ = id_
        self.event = event
        self.entities = entities
    
    def __str__(self):
        return str(self.__dict__)
    
    def avail_entities(self):
        return list(self.entities.keys())
    
    def get_entities(self, entities):

        if not isinstance(entities, list):
            entities = [entities]
        
        # Loop on entities
        return [(ent, self.entities[ent]) for ent in entities
                                         if ent in self.entities.keys()]


'''
def add_label_suffix(label, neg_label, suffix):
    return label if (label == neg_label) else \
                                           '{}_{}'.format(label, suffix)
'''

class Document(object):
    '''
    Annotated document
    '''    
    
    
    def __init__(self, fn_txt, fn_ann, outside, has_attr, entity_map, parse_sentences, remove_blank, subset=None, norm_text=False):
        
        self.fn_txt = fn_txt
        self.fn_ann = fn_ann
        self.fn = os.path.splitext(os.path.basename(fn_txt))[0]
        self.outside = outside
        self.has_attr = has_attr
        self.subset = subset
        self.entity_map = entity_map

        self.text = get_text(fn_txt)
        self.parse_sentences = parse_sentences
        self.remove_blank = remove_blank
        
        self.subs = None
        #print("\n\n Original:")
        #print(self.text)
        if norm_text:
            text, self.subs = norm_deident(self.text, check=True)
        else:
            text = self.text
        #print("New text:")
        #print(text)

        self.tokens = tokenize_doc(text, \
                        parse_sentences=parse_sentences, 
                        remove_blank=remove_blank)
                        
        self.char_indices = get_char_indices(self.text, self.tokens, subs=self.subs)


        self.events, self.textbounds, self.attributes = \
                                get_annotations(fn_ann, self.entity_map)




    def get_events(self, events):
        '''
        Get events matching the specified type (e.g. Alcohol, Drug, 
        Tobacco).
        
        args:
            types_: types as list of string or type as string
        
        returns:
            events: list of events with applicable type(s)
        '''
        return [evt for id_, evt in self.events.items() \
                 if (isinstance(events, list) and evt.event in events) \
                                              or (evt.event == events)]
        
    def get_attributes(self, textbound):
        
        # Get applicable attributes
        attributes = [attr for id_, attr in self.attributes.items() \
                                         if attr.textbound == textbound]
        
        # Number of attributes per text bound
        attr_count = len(attributes)
        
        # Make sure there is at most one attribute per textbound
        assert attr_count <= 1, \
                     "multiple attributes found:\t{}".format(attributes)
        
        # If no attributes, return dummy attribute        
        if attr_count == 0:
            return Attribute(None, None, None, None)
        
        # If only one attribute, return attribute
        else:
            return attributes[0]


    def get_tok_lab(self, start, end, event, entity):
        '''
        Get labels for token
        '''
        # Default token label
        token_label = self.outside

        # Loop on events for applicable types_
        for evt in self.get_events(event):
            
            # Loop on entities
            for ent, tb_id in evt.get_entities(entity):
                
                    # Textbound associated with entity
                    tb = self.textbounds[tb_id]
                    
                    # Look for intersection
                    if intersect(start, end, tb.start, tb.end):

                        # Use attribute value as label
                        if ent in self.has_attr:
                            attr = self.get_attributes(tb.id_)
                            new_token_label = attr.value
                        
                        # Use entity as label
                        else:
                            new_token_label = ent

                        # Current token label is negative label,
                        # so update to current
                        if token_label == self.outside:
                            token_label = new_token_label
                        
                        # No label change required
                        elif token_label == new_token_label:
                            pass
                        
                        # Make sure non-negative token
                        # label not overwritten
                        else:
                            msg = '''Duplicate token label:
                                tok={},
                                neg_label={}, 
                                token_label={}, 
                                new_token_label={}'''.format( \
                                tok.token, self.outside, token_label,
                                new_token_label)
                            raise ValueError(msg)
        return token_label
    
    def get_tok_labels(self, event, entity, binarize=False):
        '''
        Get labels for each token
        
        args:
        
        returns:
            doc_labels: document labels as list of list of string
                        (single label per token/position)
        '''        
        # Initialize note labels
        doc_labels = []
        
        # Loop on sentences
        for sent in self.char_indices:

            # Initialize sentence labels         
            sent_labels = []
            
            # Loop on tokens in current sentence
            for start, end in sent:
                
                # Label for current token
                label = self.get_tok_lab(start, end, event, entity)
               
                # Labels for sequence
                sent_labels.append(label)

            # Labels for document
            doc_labels.append(sent_labels)

        # Convert to binary labels
        if binarize:
            doc_labels = binarize_doc(doc_labels, self.outside, \
                                                       reduce_dim=False)

        return doc_labels    


    def get_tok_labels_multi(self, events, entities):
        '''
        Get multiple labels for each token
        '''
        # Initialize labels as empty list
        labels_list = [[ [] for tok in sent] 
                                          for sent in self.tokens]
        
        # Convert to list if not list
        if not isinstance(events, list):
            events = [events]
        if not isinstance(entities, list):
            entities = [entities]
        
        
        # Get labels
        for event in events:
            for entity in entities:

                # Get labels for current event-entity combination                
                labels_temp = self.get_tok_labels(event, entity)
                
                # Append labels
                for i, sent in enumerate(labels_list):
                    for j, tok in enumerate(sent):
                        labels_list[i][j].append(labels_temp[i][j])

        return labels_list


    def get_sent_labels(self, events, entities, \
                                             binarize = False, 
                                             hierarchy = None):
        '''
        Get sentence-level labels (single label per sentence)
        '''  
        
        # Get list of labels for each token
        labels_list = self.get_tok_labels_multi(events, entities)

        # Loop on sentences
        labels = []        
        for sent in labels_list:
            
            # Flatten 
            sent_flat = [lab for tok in sent for lab in tok]
            
            # Convert to binary labels
            if binarize:
                sent_lab = binarize_seq(sent_flat, self.outside, \
                                                     reduce_dim=True)
            
            # Merge labels based on hierarchy    
            else:            
                sent_lab = merge_labels(sent_flat, self.outside, \
                                                              hierarchy)
            
            # Append sentence label
            labels.append(sent_lab)
                
        return labels

    def get_seq_tags(self, events, entities, \
                                             binarize = False, 
                                             hierarchy = None):
        '''
        Get sequence tags, potentially merging labels from \
        multiple events/entities - building multiclass label set
        '''            
        
        # Get list of labels for each token
        labels_list = self.get_tok_labels_multi(events, entities)

        # Merge labels for each token
        labels_merged = [[-1 for tok in sent] for sent in labels_list]
        
        for i, sent in enumerate(labels_merged):
            for j, tok in enumerate(sent):
                
                # Convert to binary labels
                if binarize:
                    tok_lab = binarize_seq(labels_list[i][j], \
                                self.outside, reduce_dim=True)
                
                # Merge labels based on hierarchy    
                else:            
                    tok_lab = merge_labels(labels_list[i][j], \
                                self.outside, hierarchy)
                
                # Replace default value
                labels_merged[i][j] = tok_lab
         
        return labels_merged

    def get_labels(self, events, entities, \
                                             seq_tags = True,
                                             binarize = False, 
                                             hierarchy = None):
        '''
        Get labels, either sequence tags or sentence labels
        Return single set of labels
        '''                                                
        if seq_tags:
            return self.get_seq_tags(events, entities, \
                                                    binarize, hierarchy)
            
        else:
            return self.get_sent_labels(events, entities, \
                                                    binarize, hierarchy)


    def __str__(self):   
        
        text = str(self.text)
        events = stringify_dict(self.events)
        textbounds = stringify_dict(self.textbounds)
        attributes = stringify_dict(self.attributes)
       
        
        return "\n".join([
            'Text file:\t{}'.format(os.path.basename(self.fn_txt)),
            'Ann. file:\t{}'.format(os.path.basename(self.fn_ann)),                        
            '\nTEXT:\t', text, 
            '\nEVENTS:\t', events, 
            '\nTEXTBOUNDS:\t', textbounds, 
            '\nATTRIBUTES:\t', attributes])

    



class Corpus(object):
    
    def __init__(self, outside, has_attr, parse_sentences, remove_blank, norm_text=False):   
            

        self.outside = outside
        self.has_attr = has_attr
        self.entity_map = ENTITY_MAP_LOAD
        self.documents = []
        self.parse_sentences = parse_sentences
        self.remove_blank = remove_blank
        self.norm_text = norm_text
        
               
    
    def build(self, source):
        '''
        Build corpus from text files and annotation files,
        where annotation files are in standoff format
        '''

        # Find text and annotation files
        text_files, ann_files = get_files(source)
        file_list = list(zip(text_files, ann_files))
        file_list.sort(key=lambda x: x[1])
    
        # Loop on annotated files
        for fn_txt, fn_ann in file_list:
            
            # Make sure text and annotation filenames match
            assert filename_check(fn_txt, fn_ann), 'filename mismatch'
            
            # Create document
            doc = Document( \
                    fn_txt = fn_txt, 
                    fn_ann = fn_ann,
                    outside = self.outside,
                    has_attr = self.has_attr,
                    entity_map = self.entity_map,
                    parse_sentences = self.parse_sentences,
                    remove_blank = self.remove_blank,
                    norm_text = self.norm_text, 
                    )
           
            # Build corpus
            self.documents.append(doc)

        print("1 document import count:" ,len(self.documents))            
        event_count = 0
        tobacco_count = 0
        drug_count = 0
        for doc in self.documents:
            for id_, evt in doc.events.items():
                event_count += 1
                if evt.event == 'Tobacco':
                    tobacco_count += 1
                if evt.event == 'Drug':
                    drug_count += 1
        print("event count:",event_count)
        print("tobacco count:",tobacco_count)
        print("drug count:",drug_count)

                

    def split_data(self, train_val_test_split):
        '''
        Split data into training, tune, and test sets
        '''
        
        # Fix random seed
        np.random.seed(seed=1)
        
        # Get randomly assigned subsets
        doc_subsets = np.random.choice(SUBSETS, len(self.documents), \
                                            p=train_val_test_split)
    
        # Apply randomly assigned subsets
        for i, lab in enumerate(doc_subsets):
            self.documents[i].subset = lab


    def get_docs(self, subset):
        '''
        Get documents associated with subset
        '''
        if subset == SUBSET_DEFAULT:
            subsets = SUBSETS
        else:
            subsets = [subset]
            
        docs = [doc for doc in self.documents if doc.subset in subsets]
        
        return docs

    def get_doc_fns(self, subset):
        '''
        Get filenames for each document
        
        '''
        return [doc.fn for doc in self.get_docs(subset)]
        
    def get_sent_fns(self, subset):
        '''
        Get tokens
        '''
        fns = []
        for doc in self.get_docs(subset):
            fns.extend([doc.fn]*len(doc.tokens))
       
        return fns

    def get_text(self, subset):
        '''
        Get text (as string, not tokens)
        '''
        text = []
        for doc in self.get_docs(subset):
            text.append(doc.text)
        
        return text        

    def get_substitutions(self, subset):
        subs = []
        for doc in self.get_docs(subset):
            if doc.subs is not None:
                subs.extend(doc.subs)
            else:
                return None
        return subs


    def get_tokens(self, subset):
        '''
        Get tokens
        '''
        tokens = []
        for doc in self.get_docs(subset):
            tokens.extend(doc.tokens)
        
        return tokens

       
    def get_sent_labels(self, subset, events, entities, \
                                             binarize = False, 
                                             hierarchy = None):
        '''
        Get sentence-level labels
        '''                                              
        # Initialize note labels
        labels = []
        
        # Loop on documents                  
        for doc in self.get_docs(subset):

                # Get labels for current document
                labs = doc.get_sent_labels(events, entities, \
                                                  binarize = binarize,
                                                  hierarchy = hierarchy)
                # Append
                labels.extend(labs)
        
        return labels
                                                
                                                
    def get_seq_tags(self, subset, events, entities, \
                                             binarize = False, 
                                             hierarchy = None):
        '''
        Get sequence tags
        '''
        # Initialize note labels
        labels = []
        
        # Loop on documents                  
        for doc in self.get_docs(subset):

                # Get labels for current document
                labs = doc.get_seq_tags(events, entities, \
                                                  binarize = binarize,
                                                  hierarchy = hierarchy)
                # Append
                labels.extend(labs)
        
        return labels

    def get_labels(self, subset, events, entities, \
                                             seq_tags = True,
                                             binarize = False, 
                                             hierarchy = None):

        '''
        Get labels, either sequence tags or sentence labels
        '''                                                

        # Initialize note labels
        labels = []
        
        # Loop on documents                  
        for doc in self.get_docs(subset):
                
                # Get labels for current document
                labs = doc.get_labels(events, entities, seq_tags, \
                                                  binarize = binarize,
                                                  hierarchy = hierarchy)
                # Append
                labels.extend(labs)
        
        return labels

    def get_event(self, subset, label_defs, \
                                            to_list=True):
        '''
        Get multiple sets of labels
        
        args:
        '''
        
        labels = {}
        for lab_name, lab_params in label_defs.items():
            labels[lab_name] = self.get_labels(subset, **lab_params)
        
        if to_list:
            labels = dict_to_list(labels)
        
        
        return labels
    
    def get_events(self, subset, label_defs, \
                                            to_list = True):
        '''
        Get multiple sets of labels
        
        args:
        '''
        
        labels = {}
        for event, lab_params in label_defs.items():
            labels[event] = {}
            labels[event].update(self.get_event(subset, lab_params, \
                                                       to_list = False))
            
        if to_list:
            labels = nested_dict_to_list(labels)
            
        return labels    
    
    def get_event_counts(self, combos, path):
        '''
        Create label summary
        '''

        '''
        Get event counts for each type-entity combination
        '''        
        # All documents in corpus
        docs = self.get_docs(SUBSET_DEFAULT)
        
        # Entity-type counts
        counts = []
        
        # Loop on event-entity combinations
        for event, entity in combos: 
           
            # Find type-entity combinations
            found = [True for doc in docs \
                         for evt in doc.get_events(event) \
                            for ent2 in evt.get_entities(entity)]
                           
            # Append row
            counts.append((event, entity, len(found)))
        
        # Build data frame
        df = pd.DataFrame(counts, columns=['Event', 'Entity', 'Counts'])
        df = pd.pivot_table(df, values='Counts', index=['Entity'], \
                                                      columns=['Event'])

        events = list(set([ev for ev, en in combos]))
        check = []
        for doc in docs:
            for event in events:
                
                entity = event
                cnt_event = len([True for evt in doc.get_events(event) for ent2 in evt.get_entities(entity)])
                entity = "Status"
                cnt_status = len([True for evt in doc.get_events(event) for ent2 in evt.get_entities(entity)])
                    
                f = os.path.basename(doc.fn_txt)
                check.append((f, event, cnt_event, cnt_status))
        df_check = pd.DataFrame(check, columns=['fn', 'Event', 'Event_cnt', 'Status_cnt'])
        df_check['not_equal'] = df_check['Event_cnt'] != df_check['Status_cnt']
        df_check = df_check[df_check['not_equal']==True]

        # Save if path provided
        if path:
            fn = os.path.join(path, 'label_summary.xlsx')
            writer = pd.ExcelWriter(fn)
            df.to_excel(writer,'events')

            fn = os.path.join(path, 'label_counts_by_doc.xlsx')
            writer = pd.ExcelWriter(fn)
            df_check.to_excel(writer,'events')
        return df

    def get_label_distribution(self, path=None, \
                                        entities=None, events=None):
        '''
        Get distribution of labels        
        '''        
        
        # Loop on subsets        
        dfs = {}
        for ss in ['all'] + SUBSETS:

            if ss == SUBSET_DEFAULT:
                ss_ = SUBSETS
            else:
                ss_ = [ss]

            # Initialize counter
            occur_cnts = {}
            word_cnts = {}
            

            # Get documents associated with subset
            docs = [doc for doc in self.documents if doc.subset in ss_]

            # Loop on document in subset
            for doc in docs:
                
                # Loop on events       
                for _, evt_rep in doc.events.items():

                    # Get event
                    event = evt_rep.event
                    ents = evt_rep.entities
                
                    # Add event, if missing
                    if event not in occur_cnts.keys():
                        occur_cnts[event] = {}
                        word_cnts[event] = {}
                
                    
                    # Loop on entities associated with event
                    for entity, tb in ents.items():
                        
                        # Add entity-event combination, if missing
                        if entity not in occur_cnts[event].keys():
                            occur_cnts[event][entity] = 0
                            word_cnts[event][entity] = 0
                
                        # Increment Count
                        occur_cnts[event][entity] += 1
                        
                        # Text associated with text bound
                        text = doc.textbounds[tb].text
                        

                        # Increment word counts
                        toks = tokenize_doc(text, 
                                        parse_sentences=False,
                                        remove_blank=False) 
                        toks = [tok for sent in toks for tok in sent]

                        word_cnts[event][entity] += len(toks)
            # Format counts as data frame
            counts = []
            for evt, ents in occur_cnts.items():
                for ent, _ in ents.items():
                    oc = occur_cnts[evt][ent]
                    wc = word_cnts[evt][ent]
                    counts.append((evt, ent, oc, wc))
            cols = ['event', 'entity', 'occurrence count', 'word count']
            dfs[ss] = pd.DataFrame(counts, columns=cols) 
            dfs[ss].sort_values(['event', 'entity'], inplace=True)
            
            # Save as CSV
            fn = os.path.join(path, 'label_distribution_{}.csv'.format(ss))
            dfs[ss].to_csv(fn, index = False )
            
            
            '''
            Create pivot table with summary of occurrence counts
            '''
            if (entities is not None) and (events is not None) and \
                len(dfs[ss]) > 0:

                df_temp = dfs[ss].copy()
                include = [('entity', entities), ('event',  events)]
                exclude = []
                df_temp = filter_df(df_temp, include, exclude)

                idx = ["entity"]
                vals = ["occurrence count"]
                cols = ["event"]
                pt = pd.pivot_table(df_temp, \
                                        index=idx,
                                        values=vals, 
                                        columns=cols)    
            
                pt = pt.fillna(0)
                pt = pt.reindex_axis(entities, axis=0)
                pt = pt.reindex_axis([("occurrence count", evt) for evt in events], axis=1)
                fn = os.path.join(path, 'label_pivot_{}.csv'.format(ss))
                pt.to_csv(fn, float_format='%.0f')

            
        return dfs      
                    
    def as_string(self):
        '''
        Convert corpus to human readable string
        '''
        output = []
        for doc in self.documents:
            output.append('')
            output.append('='*72)
            output.append(str(doc))
        output = "\n".join(output)
        return output

    def labeled_as_string(self, events_def):
        '''
        Convert labeled corpus to string
        '''
        
        out = []
        tokens = self.get_tokens(subset=SUBSET_DEFAULT)
        labels = self.get_events(SUBSET_DEFAULT, events_def)
        
                                                            
        for i, (toks, labs) in enumerate(zip(tokens, labels)):
            d = {}
            d['token'] = toks
            
            labs2 = {'{}-{}'.format(evt, ent):lbs \
                                       for evt, ents in labs.items() for ent, lbs in ents.items()}
            d.update(labs2)
            
            df = pd.DataFrame(d)

            cols = list(df.columns.values)
            cols.remove('token')
            new_cols = ['token']+cols
            df = df[new_cols]
            df = df.transpose()
            out.append(str(df))
        
        return "\n\n".join(out)

def max_str_len(seq):
    return max([len(str_) for str_ in seq])


def intersect(x_start, x_end, y_start, y_end):
    '''
    Look for intersection between x and y ranges
    '''
    # Look for intersection
    x_idx = range(x_start, x_end)
    y_idx = range(y_start, y_end)
    return len(set(x_idx) & set(y_idx)) > 0


def stringify_dict(D):
    '''
    Convert dictionary to string
    '''        
    return json.dumps( \
                {id_:str(d) for id_, d in D.items()}, \
                                               sort_keys=True, indent=4)


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
            id_, entity_start_end, text = l.split('\t')
            
            # Split type and offsets
            entity, start, end = entity_start_end.split()
            
            # Convert start and stop indices to integer
            start, end = int(start), int(end)
            
            textbounds[id_] = Textbound(
                          id_ = id_,
                          entity=entity, 
                          start=start, 
                          end=end, 
                          text=text,
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


def parse_events(lines, entity_map):
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
            entities = {entity_map.get(entity, entity):typ \
                                             for entity, typ in entries}

            # Only include desired entities
            events[id_] = Event( \
                      id_ = id_,
                      event = event_type,
                      entities = entities)
            
    return events

def get_nonblank_lines(fn):
    '''
    Read non-blank lines
    '''

    # Open file
    with open(fn, 'r') as f:
        
        # Get non-blank lines
        lines = [l for l in f.read().split('\n') if len(l) > 0]

    return lines


def get_text(fn):
    '''
    Read and tokenize text file
    '''
    # Load tokenized file
    with open(fn,'r') as f:
        text = f.read()
   
    return text


def get_annotations(fn, entity_map):
    '''
    Load annotations, including taxbounds, attributes, and events
    '''
    # Get nonblank lines
    lines = get_nonblank_lines(fn)
    
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
    events = parse_events(lines, entity_map)
    
    # Get text bounds    
    textbounds = parse_textbounds(lines)
    
    # Get attributes
    attributes = parse_attributes(lines)

    return (events, textbounds, attributes)


def get_files(directory):
    '''
    Find text and annotation files
    '''
    files = lambda src, ext: \
                           sorted(glob.glob("{}/*.{}".format(src, ext)))
    
    # Text and annotation files
    text_files = files(directory, TEXT_FILE_EXT)
    ann_files = files(directory, ANN_FILE_EXT)
        
    # Check number of text and annotation files
    msg = 'Number of text and annotation files do not match'
    assert len(text_files) == len(ann_files), msg

    # Check the text and annotation filenames
    mismatches = [str((t, a)) for t, a in zip(text_files, ann_files) \
                                           if not filename_check(t, a)]
    fn_check = len(mismatches) == 0
    assert fn_check, '''txt and ann filenames do not match:\n{}'''. \
                        format("\n".join(fn_mismatches))

    return (text_files, ann_files)

def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def filename_check(fn1, fn2):
    '''
    Confirm filenames, regardless of directory or extension, match
    '''
    fn1 = get_filename(fn1)
    fn2 = get_filename(fn2)

    return fn1==fn2

def flatten(sequences):
    '''
    Flatten sequences of sequence
    '''
    return [entry for seq in sequences for entry in seq]


def merge_labels(y, neg_label, hierarchy=None):
    '''
    Merge labels for single token
    
    args: 
        y = list of labels for given token
        neg_label = negative label (outside label)
    '''

    # List of unique labels
    unique = set(y)
    #unique = list(set(y))

    # Only one unique label
    if len(unique) == 1:
        merged = list(unique)[0]
        
    # Only one unique label, other than negative label
    elif (len(unique) == 2) and (neg_label in unique):
        unique.remove(neg_label)
        merged = list(unique)[0]
    
    # Multiple labels found, so use hierarchy
    elif hierarchy is not None:
    
        # Indices of all values within hierarchy
        indices = [hierarchy.index(sf) for sf in unique]
        
        # Select status based on hierarchy
        merged = hierarchy[min(indices)]

        # Issue warning
        #msg = '\Status values selected:\n{}'.format(merged)
        #warnings.warn(msg)
        #warnings.warn("deprecated", Warning)

    
    else:
        print("lablels:", y)
        raise ValueError("too many sentence labels:\t{}".format(unique))


    return merged

'''
def merge_labels_seq(Y, neg_label, hierarchy=None):
    
    Merge labels for each token in sequence
    
    args:
        Y = list of tokens, where each token is represented by a 
            list of labels for given token
        neg_label = negative label (outside label)        
    
   
    return [merge_labels(y, neg_label, hierarchy) for y in Y]
'''

def label_join(lab, suffix, neg_label):
    return lab if (lab == neg_label) else "-".join([lab, suffix])

def label_split(lab):    
    pass



def binarize_seq(labels, neg_label, reduce_dim):

    # Convert to numpy array
    labels = np.array(labels)
    
    # Binarize
    labels = labels != neg_label
   
    # Cast as integer 
    labels = labels.astype(int)
     
    # Convert from sequence of sequence to sequence
    if reduce_dim:
        labels = int(np.any(labels, axis=0))
            
    return labels



def binarize_doc(labels, neg_label, reduce_dim):

    # Get binary document labels
    binary_labels = []
    for sent in labels:

        # Append document
        binary_labels.append(binarize_seq(sent, neg_label, reduce_dim))
            
    return binary_labels