

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import regex
from collections import namedtuple

#from utils.mimic import DEIDENT_PATTERN_NEW

#DEIDENT_PATTERN_NEW = r'_\*\*.+?\*\*_'
DEIDENT_PATTERN_NEW = r'_\*\*.*?\*\*_'


Tok = namedtuple('Tok', 'token start end')

NUMBER_PATTERNS = '^((\d+)|(\d+[\.\,]\d+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(ten))$'
NUMBER_TOKEN ='<num>'
NLTK_OPENING_QUOTE = r'``'
NLTK_CLOSING_QUOTE = "''"
DOUBLE_QUOTE = '"'

ABBREV_PAT = '((?<={0}[a-zA-Z0-9]{{1,3}}){0})|({0}(?=[a-zA-Z0-9]{{1,3}}{0}))'   

NUM_LIST = '(?<=^[ \t]*[\da-zA-Z]{{1,2}}){0}'
#NUM_LIST = '(?<=[\d]{{1,2}}){0}(?=[ \t])'



def pre_norm(doc, norm_pat, norm_tok, norm_tok_pad=True):
    '''
    Preprocess text to keep pattern matched string together
    '''

    # Make sure keep token not in text
    assert not bool(re.search(norm_tok, doc)),"Found keep token token"

    # Find strings, including white space, that should be kept together
    norm_matches = list(re.finditer(norm_pat, doc))

    # Replace keep matches with placeholder    
    if norm_tok_pad:   
        norm_tok = ''.join([' ', norm_tok, ' '])
    doc, norm_count = re.subn(norm_pat, norm_tok, doc)

    # Check replacement count
    assert norm_count == len(norm_matches), "Keep token sub error"
    
    return (doc, norm_matches)

def post_norm(doc, norm_matches, norm_tok):
    '''
    Postprocess text to replace keep together placeholder with
    original text
    '''
    
    # Get first match
    if len(norm_matches) > 0:
        m = norm_matches.pop(0)
    else:
        m = None

    # Iterate over sentences
    for i, sent in enumerate(doc):
        
        # Iterate over tokens and sentence
        for j, tok in enumerate(sent):
            
            # Current token contains keep token
            if norm_tok in tok:
                
                # Replace keep token
                doc[i][j] = tok.replace(norm_tok, m.group(0))

                # Get next match
                if len(norm_matches) > 0:
                    m = norm_matches.pop(0)
                else:
                    m = None
    
    return doc


def nltk_quote_correction(doc, tokenized):

    '''
    Correct NLTK substitutions for double quotes
       NLTK substitutes '``' for opening double quote
       NLTK substitutes "''" for closing double quote
       
        (r'``', '"'),                   # NLTK quotes
        ("''", '"'),                    # NLTK quotes
    '''
   
    # Document as text without white space
    doc_wo_ws = "".join(doc.split())
    
    # Iterate over sentences
    for i, sent in enumerate(tokenized):
        
        # Iterate over tokens and sentence
        for j, tok in enumerate(sent):

            # Character count
            char_count = len(tok)
            
            # Token characters do not match next characters in string
            if not doc_wo_ws[0:char_count] == tok:
                
                # Next character in string is a double quote
                next_is_quote = doc_wo_ws[0] == DOUBLE_QUOTE
                
                # Is token NLTK opening quote OR closing quote
                if (tok in [NLTK_OPENING_QUOTE, NLTK_CLOSING_QUOTE]) \
                    and next_is_quote:
                    
                    # Correct nltk substitution
                    tokenized[i][j] = DOUBLE_QUOTE
                    
                    # Adjust character count
                    char_count = len(DOUBLE_QUOTE)
                    
                # Must be an error 
                else:
                    msg = '''Token = {}
                    \nText = {}\n
                    \nFull text =\n{}'''.format(tok, doc_wo_ws, doc)
                    raise ValueError(msg)
            
            # Delete found characters from text
            doc_wo_ws = doc_wo_ws[char_count:]

    return tokenized

def tokenize_doc(doc, \
            abbrev = ABBREV_PAT,
            dummy = '__DUMMY__',
            num_list = NUM_LIST,
            parse_sentences = True,
            remove_blank = False,
            norm_pat = None,
            norm_tok = '__KEEP__',
            norm_tok_pad = True):
    '''
    Parse document into sentences and tokenize, using NLTK
    NOTE: preprocessing and postprocessing used to avoid splitting
        leading token in lists and abbreviations with periods
    
    Args:
        doc = document as string
        return_ind: 
                if False, returns doc as list of 
                    sentences where sentences are lists of tokens
                if True, returns doc as list of list of
                    namedtuple that includes the token and start 
                    and stop indices
    '''

    # Get document without white space for check at end
    doc_wo_ws = "".join(doc.split())

    # Replace keep together text with placeholder    
    if norm_pat is not None:
        doc, norm_matches = \
                         pre_norm(doc, norm_pat, norm_tok, norm_tok_pad)   

    # Make sure dummy not in text
    assert not bool(re.search(dummy, doc)), "Found dummy token"        
    
    # Preprocessing for sentence boundary detection and tokenization
    sent_preprocess = [
        (num_list.format('\.'), dummy),   # Numbered lists
        (abbrev.format('\.'), dummy), # Abbreviations
        ('\/', ' / '), # Slashes
        ('\-', ' - '), # Hyphens
    ]

    # Post processing for sentence boundary detection and tokenization
    token_postprocess = [
        (dummy, '.'), # Any dummy substitution, 
                      #     including numbered lists and abbreviations
    ]

    # Split on lines
    tokenized = doc.split('\n')

    # Preprocess line before sentence boundary detection
    for pattern, repl in sent_preprocess:
        tokenized = [regex.sub(pattern, repl, line) for line in tokenized]

    # Parse into sentences
    if parse_sentences:
        tokenized = [sent for line in tokenized \
                                        for sent in sent_tokenize(line)]
    # Tokenize sentences
    tokenized = [word_tokenize(sent) for sent in tokenized]

    # Remove blank sentences
    if remove_blank:
        tokenized = [sent for sent in tokenized if len(sent) > 0]

    # Postprocess tokens
    for pattern, repl in token_postprocess:
        tokenized = [" ".join(sent) for sent in tokenized]
        tokenized = [regex.sub(pattern, repl, sent) for sent in tokenized]
        tokenized = [sent.split() for sent in tokenized]

    # Correct NLTK quotes change
    tokenized = nltk_quote_correction(doc, tokenized)

    # Replace keep together placeholder with original text
    if norm_pat is not None:
        tokenized = post_norm(tokenized, norm_matches, norm_tok)   

    # Check characters before and after tokenization
    tok_wo_ws = "".join([tok for sent in tokenized for tok in sent])
    tok_wo_ws = "".join(tok_wo_ws.split())
    assert doc_wo_ws == tok_wo_ws, "tokenization error"

    return tokenized




def find_text_subset(full, partial):
    
    
    
    # Get text without white space
    full_no_ws = "".join(full.split())
    partial_no_ws = "".join(partial.split())
    
    # Find partial in full
    start_no_ws = full_no_ws.find(partial_no_ws)
    end_no_ws = start_no_ws + len(partial_no_ws)
    assert start_no_ws >= 0, "Could not find match"    

    i_no_ws = 0
    start_full = -1
    end_full = -1
    for i_full, char in enumerate(full):

        # Character is not whitespace
        if not char.isspace():
            
            assert full_no_ws[i_no_ws] == full[i_full], \
            '''character mismatch {} {}'''.format( \
            full_no_ws[i_no_ws], full[i_full])
            
            # Found index of beginning of overlap
            if i_no_ws == start_no_ws:
                start_full = i_full

            # Found index of end of overlap
            if i_no_ws == end_no_ws:
                end_full = i_full
            
            # Increment non-white space count            
            i_no_ws += 1

    assert start_full >= 0, "could not find start"        
    assert end_full >= 0, "could not find end"

    return (start_full, end_full)


def get_char_indices(text, tokenized, subs=None):
    
    '''
     Get token character indices
    '''
    
    if subs is not None:
        subs = subs[:]
    
    # Initialize stop index (index of last character found)
    idx_end = 0
    
    # Loop on tokenized document
    doc_indices = []
    
    # Loop on sentences in document
    for sent in tokenized:
        
        # Initialize new sentence
        sent_indices = []
        
        # Loop on tokens and sentence
        for tok in sent:

            tok_tmp = tok

            if (subs is not None) and \
               re.search(DEIDENT_PATTERN_NEW, tok):

                orig, new_ = subs.pop(0)

                # Make sure substituted value matches current token
                assert new_ in tok, '''string mismatch:
                tok = {}
                new_ = {}'''.format(tok, new_)
                    
                tok_tmp = tok_tmp.replace(new_, orig, 1)

            
            idx_start = text.index(tok_tmp, idx_end)
            skipped = text[idx_end:idx_start]
            assert (len(skipped) == 0) or (skipped.isspace()), '''
            Tried to skip: "{}"'''.format(skipped)
            idx_end = idx_start + len(tok_tmp)
            
            
            
            # Package as named tuple and depend
            sent_indices.append((idx_start, idx_end))
    
        doc_indices.append(sent_indices) 

    if subs is not None:
        assert(len(subs)) == 0, "substitutions remaining"    
    
    return doc_indices

def map_num_token(tok, num_token=NUMBER_TOKEN):
    '''
    Map numbers to special number token
    '''
    return re.sub(NUMBER_PATTERNS, num_token, tok)
    

def map_num_line(line, num_token=NUMBER_TOKEN):
    '''
    Map sequence of tokens to number token
    '''
    return [map_num_token(tok, num_token) for tok in line]
    

def map_num_doc(doc, num_token=NUMBER_TOKEN):
    '''
    Map sequence of sequence (document) of tokens to number token
    '''
    return [map_num_line(line, num_token) for line in doc]



