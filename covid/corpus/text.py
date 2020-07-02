import re
from collections import Counter
import pandas as pd
import pickle
import zlib
import json
import logging

from corpus.tokenization import tokenize_doc, NUM_LIST_PAT, INITIAL_CAP_PAT
from corpus.alignment import align
from constants import *

# Pattern for extracting sections
#   Assumptions:
#       Sections start with new line
#       Section headings include letters, forward slash, backslash, or &
#       Section headings and with a colon
SECTION_GENERIC = '^[a-zA-Z\/\\\&\? \t]+:'

DT = [ \
   'lives? *(with|alone|\/works)?', '(living|family) *(situation|\/support)? *(at home|prior to admit)?', 'hous(e|ing)', 'homes? *(environment|situation)?', 'residences?', 'support/family', 'homeless', 'home set ?up', 'marital', 'spouse name',
   '(street (or|and) illicit|illicit|rec(reational)?|other|iv)? *(drugs?|(poly ?)?substances?|ivdu|thc) *(abuse|use)? *(history|hx)?', 
   'illicits?', 'caffeine', 'coffee', 'heroin', 'amphetamines?', 'marijuana', 'cocaine', 'crack', 'methamphetamines?', '(known )?dependence',
   '(smokeless|former|previous|other|chewing)? *(tobacco|tob|smok(er?|es?|ing|ed)|cigarettes?|cigarettes?\/day|cigs|smoking\/tobacco) *(history|hx|abuse|use)?', 'cigars?', 'packs?/day',
   '(alcohol|etoh|alc|ethanol) *(history|hx|abuse|use)?',
   'family', 'married', 'marital status', 'mother', 'mom', 'wife', 'husband', 'father', 'dad', 'mother\/father', 'father\/mother', 'brothers?', 'sisters?', 'contacts?', 
   '(key)? *relationships?', 'daughters?', 'sons?', 'guardians?', 
   'm', 'f',
   '(occupational)? *exposures?', 'pets?',
   '(quit)', 'use', 'other', 'race',    
   'occ', 'occupations?', '(education)?[ /]*employment', 'work(s|ing|ed)?', 'jobs?', 'education', 'vocation(al)?', 'avocation(al)?', 'hobby', 'hobbies', 'recreational',
   '(level of)? *(activ(e|ity)|exercises?) *(level)?', 'mobility *(devices?)?', '(average)? *daily living',
   'ejection fraction', 'ef', 'patient unable to provide history', ' patient admitted from', 'wt', 'weight', 
   ]

NOT_SECTIONS = '^[ \t]*(' + '|'.join(DT) + ')[ \t]*:'

# Pattern for numbered lists

SOCIAL_PAT = '^[ \t]*((other|past|pertinent)? *(family\/|pertinent)? *(social|soc) *(and)? *(family)? *(history|hx)? *(details)?|shx?|sochx?|habits?)[ \t]*:'

PE = 'pe'
HPI = 'hpi'
HOME_MEDS = 'home meds'
PSH = 'psh'
PMH = 'pmh'
FH = 'fh'
ROS = 'ros'
MEDS = 'meds'
ALLERGIES = 'allergies'
POH = 'poh'
RT = 'results'
VITALS = 'vitals'
LB = 'lab'
CT = 'contact'
HS = 'home_service'
GE = 'general'
HC = 'hospital_course'
TP = 'temp'
CHIEF_COMPLAINT = 'cc'
ASSESSMENT = 'assess'
CONST = 'constitutional'
INJURIES = 'injuries'
ISSUES = 'issues'
SECTION_MAP = [ \
    (SOCIAL_PAT, SOCIAL),
    ('^[ \t]*((brief)? *(history)? *(of)? *present illness|hpi)[ \t]*:', HPI),
    ('^[ \t]*((admission|initial)? *(physical)? *(examination|exam) *(on|prior to)? *(presentation|admission)?|pe)[ \t]*:', PE),
    ('^[ \t]*((home)? *(medications|meds) *(at home)?)[ \t]*:', HOME_MEDS),
    ('^[ \t]*((current|admissions?|admitting|scheduled)? *(medications|meds) *(on|upon| prior to)? *(admission|presentation)?)[ \t]*:', MEDS),
    ('^[ \t]*((past)? *surg(ery|ical) *(history|hx)|pshx?)[ \t]*:', PSH),
    ('^[ \t]*((other)? *(past)? *medical *(history|hx)|pmhx?)[ \t]*:', PMH),
    ('^[ \t]*((family|fam) *(medical)? *(history|hx)|fhx?)[ \t]*:', FH),
    ('^[ \t]*(review of (sys?tems?|symptoms?)|ros)[ \t]*:', ROS),
    ('^[ \t]*(allergies)[ \t]*:', ALLERGIES),
    ('^[ \t]*((past|summary of)? *oncologic *(history|hx))[ \t]*:', POH),
    ('^[ \t]*((pertinent)? *results?)[ \t]*:', RT),
    ('^[ \t]*(vitals? *(signs?)? *(on|upon)? *(admission)?|vs)[ \t]*:', VITALS),
    ('^[ \t]*((pertinent)? *(labs?|laborator(y|ies)) *(results?|stud(y|ies)|values?|findings?|data)? *(on|upon)? *(admission|presentation)?)[ \t]*:', LB),
    ('^[ \t]*(contacts? *(persons?)? *(for|upon)? *discharge|discharge contacts?)[ \t]*:', CT),
    ('^[ \t]*((home)? *(care)? *services?)[ \t]*:', HS),
    ('^[ \t]*(gen(eral)?)[ \t]*:', GE),
    ('^[ \t]*((brief|summary of)? *hospital course)[ \t]*:', HC),
    ('^[ \t]*(temp(erature)?)[ \t]*:', TP),
    ('^[ \t]*((chief complaint)|(cc))[ \t]*:', CHIEF_COMPLAINT),
    ('^[ \t]*((assessment)( and plan)?)[ \t]*:', ASSESSMENT),
    ('^[ \t]*(injuries)[ \t]*:', INJURIES),    
    ('^[ \t]*(issues)[ \t]*:', ISSUES),        
    ('^[ \t]*(constitutional *(symptoms)?)[ \t]*:', CONST),
    ]

SECTION_START = 'START'

def wrap_sent(sent, max_seq_len):

    if max_seq_len is None:
        return [sent]    
    else:
        new_sents = []
        for j in range(0, len(sent), max_seq_len):
            new_sents.append(sent[j:j+max_seq_len])    

        # Check result
        new_sents_flat = [tok for sent_ in new_sents for tok in sent_]

        assert len(sent) == len(new_sents_flat), "token count mismatch"
        for x, y in zip(sent, new_sents_flat):
            assert x == y, "token mismatch"    
        
        return new_sents

def wrap_sentences(doc, max_seq_len):
    '''
    Wrap sentences
    '''
     
    new_doc = []
    for sent in doc:
        new_doc.extend(wrap_sent(sent, max_seq_len))    
    
    # Check truncation/wrapping
    doc_flat = [tok for sent in doc for tok in sent]
    new_doc_flat = [tok for sent in new_doc for tok in sent]
    assert len(doc_flat) == len(new_doc_flat), "length mismatch"
    
    for x, y in zip(doc_flat, new_doc_flat):
        assert x == y, "token mismatch"
    
    return new_doc


def get_indices(full_text, section_text, norm_pat=None, start=0):
    '''
    Get token indices
    '''
    # Tokenized document as list of list of string
    section_tokens = tokenize_doc(section_text, norm_pat = norm_pat)
    
    # Start and stop indices of tokens
    indices = align(full_text, section_tokens, start=start)

    return indices


def get_tokens(text, indices):
    '''
    Extract tokens from text using character indices
    '''     
       
    document = []
    for sent_indices in indices:
        sent = []
        for start, stop in sent_indices:
            sent.append(text[start:stop])
        document.append(sent)
       
    return document

def get_substitutions(orig, norm, flatten=False):
    '''
    Get substitutions
    '''
    
    # Flatten documents
    if flatten:
        orig = [tok for sent in orig for tok in sent]
        norm = [tok for sent in norm for tok in sent]
    
    # Get all substitutions
    subs = [(o, n) for o, n in zip(orig, norm) if o != n]

    # Get counts
    subs = Counter(subs)
    subs = [(o, n, c) for (o, n), c in subs.items()]

    # Package as data frame
    df = pd.DataFrame(subs, columns=['orig', 'new', 'count'])
    df.sort_values('count', inplace=True, ascending=False)
    
    return df


def get_sections(note, \
                section_generic = SECTION_GENERIC, 
                not_sections = NOT_SECTIONS, 
                num_list_pat = NUM_LIST_PAT,
                remove_linebreaks = False,
                norm_pat = None,
                keep_initial_cap = False):
    '''
    Parse document into sections
        note: document as string
    '''

    # Regular expressions for generic and target headings
    regex_generic = re.compile(section_generic, flags=re.IGNORECASE)
    regex_not_sections = re.compile(not_sections, flags=re.IGNORECASE)
    
    # Compile regular expressions for section names
    regex_sections = []
    for o, n in SECTION_MAP:
        regex = re.compile(o, flags=re.IGNORECASE)
        regex_sections.append((regex, n))          
    
    # Split into lines (not sentences)
    lines = note.splitlines()
    
    # Find indices of headings
    section_indices = []
    section_names_orig = []
    section_names_norm = []
    for i, text in enumerate(lines):
        
        # Search for section heading in line
        section_match = regex_generic.match(text)
        not_section_match = regex_not_sections.match(text)
        
        # If match found, get index and section name
        if section_match and not not_section_match:

            # Append index of current line
            section_indices.append(i)     
            
            # Append the name of current section     
            sect_name = section_match.group()
            section_names_orig.append(sect_name)
            for regex, n in regex_sections:
                if regex.match(text):
                    sect_name = n
                    break
            section_names_norm.append(sect_name)
            
            
    # Include first line, if not included
    if 0 not in section_indices:
        section_indices.insert(0, 0)
        section_names_orig.insert(0, SECTION_START)
        section_names_norm.insert(0, SECTION_START)

    # Start and stop indices for section    
    start = section_indices[:]
    stop = start[1:] + [len(lines)]
    
    # Loop on sections
    note_text = []
    note_sections_orig = []
    note_sections_norm = []
    note_indices = []
    start_char = 0
    assert len(section_names_orig) == len(start), "length mismatch"
    assert len(section_names_norm) == len(start), "length mismatch"
    for name_orig, name_norm, st, sp in zip(section_names_orig, section_names_norm, start, stop):
        
        # Section text
        section_text = "\n".join(lines[st:sp])
        if remove_linebreaks:
            section_text = rm_linebreaks(section_text, num_list_pat, keep_initial_cap=keep_initial_cap)           
        note_text.append(section_text)
        
        # Section token indices
        section_indices = get_indices(note, section_text, norm_pat, start=start_char)
        start_char = section_indices[-1][-1][1]
        note_indices.extend(section_indices)
        
        # Section names by sentence
        note_sections_orig.extend([name_orig]*len(section_indices))
        note_sections_norm.extend([name_norm]*len(section_indices))
    
    # Check extraction of sections
    orig = ''.join(note.split())
    new_ = ''.join("".join(note_text).split())
  
    assert orig == new_, \
        '''error in extracting sections
                char length string = {}
                char length sectioned = {}
        '''.format(len(orig), len(new_))
    
    from_idx = ''.join([note[idx[0]:idx[1]] for sent_idxs in note_indices for idx in sent_idxs])                            
    from_idx = ''.join(from_idx.split())
    assert orig == from_idx, \
        '''error in extracting token indices
                char length string = {}
                char length sectioned = {}
                orig =\n{}
                new = \n{}
        '''.format(len(orig), len(from_idx), orig, from_idx)

    assert len(note_indices) == len(note_sections_orig), 'length mismatch'
    assert len(note_indices) == len(note_sections_norm), 'length mismatch'

    return (note_indices, note_sections_norm, note_sections_orig)


def rm_linebreaks(doc, \
                num_list_pat = NUM_LIST_PAT, 
                section_pat = SECTION_GENERIC,
                initial_cap_pat = INITIAL_CAP_PAT, 
                keep_initial_cap = False):
    '''
    Remove linebreaks and use sentence boundary detection
    '''
    
    if (num_list_pat is None) and (section_pat is None) and (initial_cap_pat is None):
        new_doc = re.sub('\n', ' ', doc)
    else:
        # Regular expression for matching numbered lists and
        # section headings       
        num_list_regex = re.compile(num_list_pat, flags=re.I)
        section_regex = re.compile(section_pat, flags=re.I)
        initial_cap_regex = re.compile(initial_cap_pat)

        # Split into lines (not sentences)
        lines = doc.splitlines()

        # Create new document without extra linebreaks
        new_doc = []
        for ln in lines:
            
            if num_list_regex.search(ln) or \
               section_regex.search(ln) or \
               ((keep_initial_cap == True) and (initial_cap_regex.search(ln))) :
                sep = '\n' 
            else:
                sep = ' '

            new_doc.append(sep)           
            new_doc.append(ln)
            
        new_doc = "".join(new_doc)

    return new_doc         

def compare_dims(X, Y):
    '''
    Compare dimensions of list of lists
    '''
    check = []
    check.append(len(X) == len(Y))
    for x, y in zip(X, Y):
        check.append(len(x) == len(y))
    return all(check)

def compare_tokens(X, Y):
    '''
    Compare list of lists for equivalency
    '''
    assert compare_dims(X, Y), "length error"
    
    check = []
    for x, y in zip(X, Y):
        check.append(x == y)
    return all(check)    


def compare_dims_labels_multi(labels, X):

    check = []
    
    # Check dimensionality before saving
    check.append(len(X) == len(labels))
    
    # Loop on sentences
    for x, labs in zip(X, labels):
        
        # Skip lab values of None
        if labs is not None:
            
            # Loop on events
            for event, entities in labs.items():
                
                # Loop on entities
                for entity, lab in entities.items():
                    
                    # Compare length of sequence labels
                    if isinstance(lab, list):
                        check.append(len(x) == len(lab))
    return all(check)


def divisors(x):
    
    if x < 1:
        return None
    
    y = []
    for i in range(1, x):
        if x % i == 0:
            y.append(i)
    
    return y
        
def remove_duplicates(X):
    
    # Sequence count
    n = len(X)
    
    # Exit if len 0 or 1
    if n <= 1:
        return X

    # Get divisors in descending order
    D = divisors(n)

    # Loop on divisors
    for d in D:
        
        # Get first subset
        s0 = X[0:d]
        
        # Compare with remaining subsets
        equiv = False
        for s in range(d, n, d):

            # Next subset for comparison
            sn = X[s:s+d]

            # Compare subsets            
            equiv = sn == s0
            
            # Quit looking on remaining subsets
            if not equiv:
                break
        
        # Found shortest repeating subsequence
        if equiv:
            # Check result
            assert n % d == 0
            assert s0*int(n/d) == X
            
            logging.info('')
            logging.info('')
            logging.info("Duplicate found")
            logging.info('')
            logging.info("Original sequences:")
            for x in X:
                logging.info(x)
            logging.info('')                
            logging.info("Unique sequences:")
            for s in s0:
                logging.info(s)
            logging.info('')
            
            return s0
    
    # No repeating, so return original sequence
    return X            
        