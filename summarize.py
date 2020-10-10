import numpy as np
import pandas as pd

# %%
def df_obj(df):
    """
    Returns the df only containing object columns
    Paramter: df
    """
    mask = np.array(df.dtypes == 'object')
    obj_df = df.iloc[:, mask]
    return obj_df

# %%
def object_cols (df):
    """
    Returns the object columns of the dataframe and their unique values (don't include NaN in counts). 
    Paramter: df
    """
    obj_df = df_obj(df)
    obj_cols = pd.DataFrame(obj_df.dtypes, columns=['dtypes'])
    obj_cols['unique_values'] = obj_df.nunique()
    return obj_cols