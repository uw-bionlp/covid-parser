import json
import re

MILLNAMES = ['',' K',' M',' B',' T']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(MILLNAMES)-1,
                int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), MILLNAMES[millidx])

def dict_to_list(D):
    
    # Check dimensions
    len_dict = {k:len(v) for k, v in D.items()}
    lengths = [v for _, v in len_dict.items()]
    assert len(set(lengths)) == 1, "length mismatch: {}".format(len_dict)
    length = lengths[0]

    # Build list of dictionaries
    L = []
    for i in range(length):

        # Loop on dictionary
        d = {k:v[i] for k, v in D.items()}
            
        # Append to list
        L.append(d)
    return L
    
def list_to_dict(L, extend_or_append='append'):
    
    # Initialize dictionary
    D = {k:[] for k, v in L[0].items()}
    
    # Loop on elements in list
    for l in L:
               
        # Append values from current element
        for k, v in l.items():
            
            if extend_or_append == 'append':
                D[k].append(v)
            elif extend_or_append == 'extend':
                D[k].extend(v)
            else:
                raise ValueError('incorrect value for extend_or_append')
    return D


def nested_dict_to_list(D):
    
    # Check dimensions
    len_dict = {evt: {ent: len(labs) for ent, labs in ents.items()} \
                                             for evt, ents in D.items()}
        
        
    lengths = [ln for evt, ents in len_dict.items() \
                                          for ent, ln in ents.items()]
    assert len(set(lengths)) == 1, "length mismatch: {}".format(len_dict)
    length = lengths[0]

    # Build list of dictionaries
    L = []
    for i in range(length):

        # Loop on dictionary
        d = {evt:{ent:labs[i] for ent, labs in ents.items()} \
                                          for evt, ents in D.items()}
            
        # Append to list
        L.append(d)
    return L


def list_to_nested_dict(X):
    
    # Initialize dictionary
    D = {evt:{ent:[] for ent, labs in ents.items()} \
                                          for evt, ents in X[0].items()}   
    # Loop on elements in list
    for x in X:
               
        # Append values from current element
        for evt, ents in x.items():
            for ent, labs in ents.items():
                D[evt][ent].append(labs)
    return D

    
def tuple2str(entity_type):
    return "_".join(list(entity_type))

def str2tuple(entity_type):
    return tuple(entity_type.split('_'))

def stringify_dict_keys(D):
    return {tuple2str(k): v for k, v in D.items()}

def tuplify_dict_keys(D):
    return {str2tuple(k): v for k, v in D.items()}

def flatten(X):
    return [x_ for x in X for x_ in x]
    
    
def print_section(s):
    print('\n'*2)
    print('='*72)
    print('= ' + s)   
    print('='*72)    

def print_subsection(s):
    print('\n'*2)
    print('-'*72)
    print('- ' + s)   
    print('-'*72)      

def stringify_dict(D):
    '''
    Convert dictionary to string
    '''        
    return json.dumps( \
                {id_:str(d) for id_, d in D.items()}, \
                                               sort_keys=True, indent=4)


def to_flat_list(X):
    '''
    Convert argument to list
    '''    
    
    # Accommodate multiple argument types
    if isinstance(X, list):
        
        # Nested list (doc) of event
        if isinstance(X[0], list):
            return [x_ for x in X for x_ in x]
        
        # List (sent) of event
        else:
            return X
    
    # Argument is single event, so convert to list
    else:
        return [X]
            
    
def get_descrip(dir_, \
        multiple_runs = True, 
        descrip_pat = '(.*)_[0-9]+$',
        run_pat = '.*?([0-9]+)$'
        ):



    if multiple_runs:
        descrip_match = re.match(descrip_pat, dir_)    
        descrip = descrip_match.group(1) if descrip_match else dir_
            
            
        run_match = re.match(run_pat, dir_)    
        run = run_match.group(1) if run_match else 0
    else:
        descrip = dir_
        run = 0    
        
    return (descrip, run)