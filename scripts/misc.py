import numpy as np
import pandas as pd


def add_metadata_to_multiindex(df, df_meta):
    """Takes a df of count data and adds metadata as a column multiindex.
    
    The two input frames must have the same column names.
    
    Parameters
    ----------
    df : pd.DataFrame, count data with m rows, n columns
    df_meta : pd.DataFrame, metadata with k rows, n columns
    """
    
    if np.all(df.columns != df_meta.columns):
        raise Exception("df and df_meta must have matching columns")
        
    if "Sample" in df_meta.index:
        warnings.warn("df_meta has a row named 'Sample',  \
                      this will result in a duplicated multiindex")
    
    col_arrays = np.vstack([df_meta.values, df.columns.to_numpy()])
    multi_cols = pd.MultiIndex.from_arrays(col_arrays, 
                                           names=df_meta.index.to_list()
                                           + ["Sample"])
    
    return pd.DataFrame(df.values, index=df.index, columns=multi_cols)