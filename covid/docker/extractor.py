import json
import os
import logging

from constants import *
from corpus.text import wrap_sentences, get_sections, get_tokens
from corpus.document import Document

from pytorch_models.pretrained import load_pretrained
from pytorch_models.multitask.model import MultitaskEstimator
from pytorch_models.event_extractor.model import EventExtractor

class Extractor(object):
    
    '''
    Base classifier
    '''
    
    def __init__(self, model_type, model_dir, word_embed_dir, \
        param_map=None, json_indent=4):
        '''
        
        Parameters
        ----------
        model_type: str, model type in ('sdoh', 'covid')
        model_dir: str, directory with extraction model state 
                               dictionary and hyperparameter config file
        word_embed_dir: str, directory with word embedding model and 
                                hyperparameter config file
        param_map: dict, map for hyperparameters
        json_indent: int, indent, as character count, for json file
        '''

        self.model_type = model_type
        self.model_dir = model_dir
        self.word_embed_dir = word_embed_dir
        self.param_map = param_map
        self.json_indent = json_indent

        # Get model class
        # SDOH
        if self.model_type == 'sdoh':
            self.model_class = MultitaskEstimator
            self.section = 'social'
        
        # COVID
        elif self.model_type == 'covid':
            self.model_class = EventExtractor
            self.section = None
        
        # Invalid model type
        else:
            msg = "Invalid model_type:\t{}".format(self.model_type)
            raise ValueError(msg)        
        

        # Load pretrained model       
        self.model, self.hyperparams = load_pretrained( \
                                    model_class = self.model_class, 
                                    model_dir = self.model_dir,
                                    word_embed_dir = self.word_embed_dir, 
                                    param_map = self.param_map)


    def __str__(self):
        return str(self.model_class)
    

    def predict(self, text, device=-1, output_format='json', \
            run_qc=False):
        '''
        Apply extractor to text
        
        Parameters
        ----------
        
        text: document text, as str
        device: GPU device, as int. -1 indicates CPU
        
        '''


        # Limit sentence length by wrapping long sentences                
        max_len = self.hyperparams['max_len']
        max_len -= int(self.hyperparams['pad_start'])
        max_len -= int(self.hyperparams['pad_end'])


        # Create document object, to tokenize the input text
        doc = Document( \
                        text = text,
                        id_ = 'unknown',
                        tags = None,
                        patient = None,
                        norm_pat = None,
                        norm_map = None,
                        parse_sections = True,
                        remove_linebreaks = False,
                        keep_initial_cap = False,
                        max_len = None,
                        compress = False)

        # Get to the character indices associated with each token
        indices = doc.indices( \
                        section = self.section, 
                        max_len = max_len)

        # Get the tokenized sentences
        sents = doc.sents( \
                        section = self.section, 
                        max_len = max_len)

        # Generate predictions        
        # Extractor input is a list of documents, so nest sentences in list
        # Extractor output is a list of documents, so unnest doc from list
        events = self.model.predict(X=[sents], device=device)[0]

        # Return Python object
        if output_format == 'obj':
            return events
        
        # Return json string
        elif output_format == 'json':
        
            # Flatten output events
            events = [evt for sent in events for evt in sent]

            # Convert Event object to dictionary
            event_dicts = []
            for evt in events:
                d = evt.to_dict(char_indices=indices)
                
                if run_qc:
                    for arg in d["arguments"]:
                        i, j = arg["indices"]
                        a = arg["text"]
                        b = text[i:j]
                        assert a == b, '''"{}" vs. "{}"'''.format(a, b)
                
                event_dicts.append(d)
            
            # Generate JSON representation
            json_str = json.dumps(event_dicts, indent=self.json_indent)
            
            return json_str
        
        # Return brat string
        elif output_format == 'brat':
            msg = "BrAT output_format currently not available"
            raise ValueError(msg)            
        
        # Invalid output format
        else:
            msg = "Invalid output_format:\t{}".format(output_format)
            raise ValueError(msg)