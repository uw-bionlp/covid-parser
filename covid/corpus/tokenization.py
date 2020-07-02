import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import re
import regex
from collections import namedtuple

DOUBLE_QUOTE = '"'

#PUNCT_PAT = r'\?+'
PUNCT_PAT = '(\?+)'
PUNCT_PH = '__PUNCT__'

ABBREV_PAT = '(([0-9]{0,2}[a-zA-Z]{1,2}[0-9]{0,2}\.){2,4})'
ABBREV_PH = '__ABBV__'

#NUM_LIST_PAT = '^[ \t]*(([\d]{1,2}|[A-Za-z]) ?\.| *-)'
NUM_LIST_PAT = '^[ \t]*(([\d]{1,2}|[A-Za-z])\.) ?'
NUM_LIST_PH = '__NUML__'

QUOTE_PAT = '''((")|(''))'''
QUOTE_PH = '__QOTE__'

#PUNCT_WS = '([-/])'
PUNCT_WS = '([-/+])'

NORM_PH = '__NORM__'


INITIAL_CAP_PAT = '^[A-Z]'

def normalize(doc, pat, ph):
    '''
    Preprocess text to keep pattern matched string together
    '''

    if pat is None:
        return (doc, [])

    # Make sure keep token not in text
    assert not bool(re.search(ph, doc)),"Found keep token token"

    # Find strings, including white space, that should be kept together
    #matches = [m.group(0) for m in re.finditer(pat, doc, flags=re.MULTILINE)]
    matches = [m.group(1) for m in re.finditer(pat, doc, flags=re.MULTILINE)]

    # Replace keep matches with placeholder    
    ph = ''.join([' ', ph, ' '])
    
    # Replace text
    doc, norm_count = re.subn(pat, ph, doc, flags=re.MULTILINE)
    
    # Check replacement count
    assert norm_count == len(matches), "Keep token sub error"
        
    return (doc, matches)

def restore(doc, matches, ph):
    '''
    Postprocess text to replace keep together placeholder with
    original text
    '''

    # Exit early if no matches
    if len(matches) == 0:
        return doc

    # Iterate over sentences
    for i, sent in enumerate(doc):
        
        # Iterate over tokens in sentence
        for j, word in enumerate(sent):

            # Current token contains keep token
            if ph == word:
                
                # Replace keep token
                doc[i][j] = matches.pop(0)
    
    # Confirm all matches consumed
    assert len(matches) == 0, "Not all matches consumed"
    
    return doc

def tokenize_doc(text, \
            norm_pat = None,
            abbrev_pat = ABBREV_PAT,
            num_list_pat = NUM_LIST_PAT,
            quote_pat = QUOTE_PAT,
            #punct_pat = PUNCT_PAT,
            punct_ws = PUNCT_WS):
    '''
    Parse document into sentences and tokenize, using NLTK
    NOTE: preprocessing and postprocessing used to avoid splitting
        leading token in lists and abbreviations with periods
    
    Args:
        text = document as string
        return_ind: 
                if False, returns text as list of 
                    sentences where sentences are lists of tokens
                if True, returns text as list of list of
                    namedtuple that includes the token and start 
                    and stop indices
    '''

    # Get document without white space for check at end
    text_wo_ws = "".join(text.split())
           
    # Normalize text before tokenization
    norm_list = [ \
        (norm_pat,      NORM_PH), 
        (abbrev_pat,    ABBREV_PH), 
        (num_list_pat,  NUM_LIST_PH), 
        (quote_pat,     QUOTE_PH),
        #(punct_pat,     PUNCT_PH)
        ]
    matches = []    
    for pat, ph in norm_list:
        text, mat = normalize(text, pat, ph)
        matches.append((mat, ph))

    # Pad punctuation with white space
    text = re.sub(punct_ws, r' \1 ', text)

    # Split on lines
    lines = text.split('\n')

    # Parse into sentences
    tokenized = [sent for line in lines for sent in sent_tokenize(line)]
    
    # Tokenize sentences
    tokenized = [word_tokenize(sent) for sent in tokenized]

    # Remove blank sentences
    tokenized = [sent for sent in tokenized if len(sent) > 0]

    # Restore normalization on token-by-token basis
    for mat, ph in matches:
        tokenized = restore(tokenized, mat, ph)

    # Check characters before and after tokenization
    tok_wo_ws = "".join([tok for sent in tokenized for tok in sent])
    tok_wo_ws = "".join(tok_wo_ws.split())
    assert text_wo_ws == tok_wo_ws, \
        '''tokenization error:
           orig:
           {}
           {}
           new:
           {}'''.format(text_wo_ws, '='*200, tok_wo_ws)

    return tokenized


