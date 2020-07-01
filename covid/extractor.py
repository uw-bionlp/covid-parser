import json

from corpus.text import wrap_sentences, get_sections, get_tokens


class Extractor(object):
    
    '''
    Base classifier
    '''
    
    def __init__(self,  model_class, model_dir, xfmr_dir, \
        json_indent = 4):
            
        # Model class            
        self.model_class = model_class
        
        # Model directory (directory with state dictionary and hyperparameter config file)
        self.model_dir = model_dir

        # Indent, as character count, for json file
        self.json_indent = json_indent

        # Instantiate model
        self.model = self.model_class()

        # Load pretrained model
        param_map={'xfmr_dir': xfmr_dir}
        self.model.load_pretrained(self.model_dir, param_map=param_map)


    def __str__(self):
        return str(self.model_class)


    def predict(self, text):
        
        
        '''
        Tokenization
        '''

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
        events_flat = [evt for doc in events for sent in doc for evt in sent]

        event_dicts = []
        for evt in events_flat:
            event_dicts.append(evt.to_dict(char_indices=indices))
        
        json_str = json.dumps(event_dicts, indent=self.json_indent)
        
        return json_str