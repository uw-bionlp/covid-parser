import pandas as pd
import numpy as np


def filter_df(df, include=None, exclude=None):
    '''
    Filter a data frame based on criteria
    
    Args:
    df = pandas dataframe
    include = list of tuples (column, value)
    '''


    # Create mask
    mask = []
    mask.append([True]*len(df))

    if include:
        for column, value in include:

            # Handle None
            if value is None:
                B = df[column].isnull().tolist()
                
            # Check if column value is in provided list
            elif isinstance(value, list) or \
                 isinstance(value, tuple) or \
                 isinstance(value, set):
                B = (df[column].isin(value)).tolist()

            # Basic Boolean check
            else:
                B = (df[column] == value).tolist()

            mask.append(B)
        
                    
    if exclude:
        for column, value in exclude:
            B = (df[column] != value).tolist()
            mask.append(B)
        
                    
    mask = np.array(mask)
    
    mask = np.all(mask, axis=0)

    return df[mask]


def best_by_group(df, sort_cols, ascending, groups, criteria=None):

    # Filter by criteria
    if criteria:
        df = filter_df(df, criteria)
    
    # Sort by column of interest
    df = df.sort_values(sort_cols, ascending=ascending)
     
    # Apply grouping 
    df = df.groupby(groups, as_index=False).first()
    
    return df
    
def best_row(df, sort_cols, ascending, criteria=None):

    # Filter by criteria
    if criteria:
        df = filter_df(df, criteria)
    
    # Sort by column of interest
    df = df.sort_values(sort_cols, ascending=ascending)
    
    # Apply grouping 
    df = df.iloc[0]
    
    return df
    
def stringify(x):    

    if isinstance(x, tuple) or isinstance(x, list):
        return " ".join(x)
    else:
        return x


def stringify_cols(df):
    
       
    df.columns = df.columns.to_series().apply(stringify)
    return df
