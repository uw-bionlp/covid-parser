import os
import json

from corpus.text import wrap_sentences, get_sections, get_tokens
from models.event_extractor.wrapper import EventExtractorWrapper

class DocumentProcessor():

    def __init__(self):
        self.model_class = EventExtractorWrapper
        self.model_dir   = os.path.join(os.getcwd(), 'models/covid/')
        self.model       = self.model_class()
        self.model.load_pretrained(self.model_dir, param_map={ 'xfmr_dir' : os.path.join(os.getcwd(), 'pretrained/biobert_pretrain_output_all_notes_150000/') })


    def predict(self, text):

        # Tokenize document        
        indices, sections, sections_orig = get_sections(text)
        indices = wrap_sentences(indices, 30)
        sents = get_tokens(text, indices)
        
        # Extractor operates on a list of documents
        sents = [sents]
        
        '''
        Prediction
        '''
        events = self.model.predict(sents, device=-1)
        
        '''
        Postprocessing
        '''
        events_flat = [ evt for doc in events for sent in doc for evt in sent ]
        return [ evt.to_dict(char_indices=indices) for evt in events_flat ]