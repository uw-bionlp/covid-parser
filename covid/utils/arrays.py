import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix




def get_subset(X, idx):
    '''
    Get subset of X based on indices listed in idx
    '''   
    
    # Numpy array
    if isinstance(X, (np.ndarray, np.generic)):
        return X[idx]
    
    # Sparse, spicy matrix
    elif isinstance(X, csr_matrix):
        return X[idx, :]
        
    # List    
    elif isinstance(X, list):
        return [X[i] for i in idx]
    
    # Cannot figure out type    
    else:
        raise TypeError("Could not coerce type")
        
def apply_mask(X, mask):
    '''
    Apply mask to sequence (list, numpy array, spicy matrix)
    '''
        
    # Indices 
    idx = np.where(np.array(mask))[0]   
    
    return get_subset(X, idx)