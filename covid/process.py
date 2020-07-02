import os
import json

from corpus.text import wrap_sentences, get_sections, get_tokens
from models.event_extractor.wrapper import EventExtractorWrapper
from models.xfmrs import get_model, get_tokenizer
from models.utils import get_device
from constants import BERT

class DocumentProcessor():

    def __init__(self):
        self.model_class     = EventExtractorWrapper
        self.model_dir       = os.path.join(os.getcwd(), 'models/covid/')
        self.model           = self.model_class()
        self.pretrained_dir  = os.path.join(os.getcwd(), 'pretrained/biobert_pretrain_output_all_notes_150000/')
        self.model.load_pretrained(self.model_dir, param_map={ 'xfmr_dir' : self.pretrained_dir })
        self.device          = get_device()

        self.embedding_model = get_model(BERT, self.pretrained_dir)
        self.embedding_model.eval()
        self.embedding_model.to(get_device())

        self.tokenizer_model = get_tokenizer(BERT, self.pretrained_dir)


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
        events = self.model.predict_fast(sents, self.embedding_model, self.tokenizer_model, device=self.device)
        
        '''
        Postprocessing
        '''
        events_flat = [ evt for doc in events for sent in doc for evt in sent ]
        return [ evt.to_dict(char_indices=indices) for evt in events_flat ]