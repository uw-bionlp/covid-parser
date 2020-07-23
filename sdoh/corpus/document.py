

from constants import START_TOKEN, END_TOKEN


from corpus.text import wrap_sentences, get_sections, get_tokens
from corpus.text import wrap_sent, normalize_doc
from corpus.text import get_indices
from corpus.text import comp_text, decomp_text, comp_indices, decomp_indices
from corpus.labels import doc_level_labels, doc_keyword_ind
from corpus.text import remove_duplicates
from corpus.brat import write_txt, write_ann, create_ann_str


from constants import *




class Document:
    '''
    Document container
    '''
    def __init__(self, \
        
        # Raw/original text
        text,
        
        # Identification and data splits
        id_,
        tags = None,
        patient = None,
        date = None,
        
        # Normalization and tokenization
        norm_pat = None,
        norm_map = None,
        parse_sections = False,
        remove_linebreaks = False,
        keep_initial_cap = False,
        max_len = None,
        
        # Storage
        compress = False, 
        

        ):



        # Identification and data splits
        self.id_ = id_
        self.tags = set([]) if tags is None else tags
        self.patient = patient
        self.date = date
        
        # Normalization and tokenization
        self.norm_pat = norm_pat
        self.norm_map = norm_map
        self.parse_sections = parse_sections
        self.remove_linebreaks = remove_linebreaks
        self.keep_initial_cap = keep_initial_cap
        self.max_len = max_len

        # Storage
        self.compress = compress

        # Original text
        self._text = comp_text(text) if compress else text.encode(ENCODING).decode(ENCODING)

        # Get token indices with section parsing
        if parse_sections:
            indices, sections, sections_orig = get_sections(text, \
                                                remove_linebreaks = remove_linebreaks,
                                                norm_pat = norm_pat,
                                                keep_initial_cap = keep_initial_cap)

        # Get token indices without section parsing
        else:
            indices = get_indices(text, text, norm_pat)
            sections = None
            sections_orig = None

        # Indices and sections
        self._indices = comp_indices(indices) if compress else indices
        self._sections = sections
        self._sections_orig = sections_orig


    '''
    ====================================================================    
    Built-in
    ====================================================================
    '''


    #def __eq__(self, other):
    #    '''
    #    Define equivalence
    #    '''
    #    
    #    assert False, "need to rework to accommodate removal of category and subcat attribute"
    #    
    #    fn_match = self.id_ == other.id_
    #    category_match = self.category == other.category
    #    subcategory_match = self.subcat == other.subcat
    #    return fn_match and category_match and subcategory_match

    

    '''
    ====================================================================    
    Get text/tokens
    ====================================================================
    '''
    
    def text(self):
        '''
        Get document text
        '''
        txt = decomp_text(self._text) if self.compress else self._text
        return txt
    
    def indices(self, section=None, max_len=None):
        '''
        Get indices of document tokens
        '''
        
        # Character indices of tokens
        indices = decomp_indices(self._indices) if self.compress \
                                                      else self._indices
        
        # Filter based on section
        if section is not None:
            
            # Section names of sentences
            sections = self._sections

            # Check lengths (sentence count)
            assert len(indices) == len(sections), \
                          '{} vs {}'.format(len(indices), len(sections))
 
            # Filter by looping on sentences
            indices = [idx for idx, sect in zip(indices, sections) \
                                                     if sect == section]                                  

        # Wrap long sentences                                                      
        if max_len is not None:
            indices = wrap_sentences(indices, max_len)
                                                      
        return indices

    def sections(self, max_len=None):
        '''
        Get section labels for each sentence, 
        including wrapping for length
        '''
        
        # Get indices unwrapped (no length limitation)
        indices = self.indices(section=None, max_len=None)
        
        sect_by_sent = []
        assert len(indices) == len(self._sections), '{} vs {}'.format(len(indices), len(self._sections))
        for ind, sect in zip(indices, self._sections):
            sent_count = len(wrap_sent(ind, max_len))
            sect_by_sent.extend([sect]*sent_count)
    
        return sect_by_sent

    def section_comp(self):
        return zip(self._sections_orig, self._sections)
            

    def sents(self, section=None, norm=False, max_len=None, \
                                rm_duplicates=False):
            
        '''
        Get tokenized document
        '''
        
        # Indices
        indices = self.indices(section=section, max_len=max_len)
        
        # Tokenize document
        sents = get_tokens(self.text(), indices)

        ## Get section subset
        #if section is not None:
        #
        #    # Get sections by sentence
        #    sections = self.sections(max_len)
        #
        #    # Filter sentences
        #    sents = section_subset(sents, sections, section)

        if rm_duplicates:
            sents = remove_duplicates(sents)
        
        # Normalize sents
        if norm:
            sents = normalize_doc(sents, self.norm_pat, self.norm_map)
       
        return sents               


    '''
    ====================================================================
    Get size
    ====================================================================
    '''            

    def sent_count(self, section=None, max_len=None):
        '''
        Sentence count
        '''
        return len(self.sents(section=section, max_len=max_len))



    '''
    ====================================================================
    Representations
    ====================================================================
    '''    

    #def doc_as_tuple(self, \
    #    norm = False,
    #    max_len = None):
    #    '''
    #    Get tuple representation of document
    #    '''
    #    assert False, "need to rework to accommodate removal of category and subcat attribute"
    #
    #    id_ = self.id_
    #    cat = self.category
    #    sub = self.subcat
    #    txt = self.text()
    #            
    #    # Get tokens    
    #    toks = self.sents( \
    #            norm = norm, 
    #            max_len = max_len)
    #
    #      
    #    return (self.id_, self.category, self.subcat, txt, toks)

    #def sent_as_tuple(self, \
    #    norm = False,
    #    max_len = None):
    #    '''
    #    Get tuple representation of document
    #    '''
    #    assert False, "need to rework to accommodate removal of category and subcat attribute"
    #            
    #    # Get tokens    
    #    toks = self.sents( \
    #            norm = norm, 
    #            max_len = max_len)
    #
    #    # Return doc as list of sentences
    #    doc = []
    #    for sent in toks:
    #        doc.append((self.id_, self.category, self.subcat, sent))
    #            
    #    return doc



    '''
    ====================================================================
    Labels
    ====================================================================
    '''    


    def add_labels_multi(self, labels, max_len):
        '''
        Incorporate multi-task labels
        '''
        
        if labels is None:
            self.max_len = None
            self.labels = None
        else:
        
            # Maximum sentence length for label
            self.max_len = max_len
            
            # Wrap indices for dimensionality check
            indices = wrap_sentences(self.indices(), max_len)
            
            
            assert compare_dims_labels_multi(labels, indices), \
                                                    "Dimensionality error"
            
            self.labels = labels
        
        return True 


    def to_brat(self, path, events, section=None, max_len=None,
                arg_map=None, tb_map=None, attr_map=None):
        '''
        Save document in brat format
        '''

        # Character indices
        char_indices = self.indices(section=section, max_len=max_len)

        # Convert tokens indices to char indices
        assert len(char_indices) == len(events), \
                       '{} vs {}'.format(len(char_indices), len(events))

        # Include character indices for tokens
        # Loop on sentences
        for sent_events in events:
            
            # Loop on events in current sentence
            for evt in sent_events:
                        
                # Loop on argument spans in current event
                for span in evt.arguments:
                        
                    # Add character indices
                    span.add_char_idxs(char_indices)
        
        
        # Create annotation string
        ann = create_ann_str(events, \
                                arg_map = arg_map, 
                                tb_map = tb_map, 
                                attr_map = attr_map)

        write_txt(path, self.id_, self.text())
        write_ann(path, self.id_, ann)

        return True 


    def get_doc_labels(self, events, entity, hierarchy):
        '''
        Get document-level labels
        '''       
        return doc_level_labels(self.labels, events, entity, hierarchy)

    def get_doc_keyword_ind(self, event, entity, label, pattern):
        '''
        Get indicator
        '''
        
        tokens = self.sents( \
            max_len = self.max_len)
        
        return doc_keyword_ind( \
            labels = self.labels, 
            tokens = tokens, 
            event = event, 
            entity = entity, 
            label = label,
            pattern = pattern)        
                 
