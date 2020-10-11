import numpy as np
import pandas as pd

# %%
def obj_df(df):
    """
    Returns the dataframe only containing object columns
    Paramter: df
    """
    mask = np.array(df.dtypes == 'object')
    df_obj = df.iloc[:, mask]
    return df_obj

# %%
def obj_cols (df):
    """
    Returns the object columns of the dataframe and their unique values (don't include NaN in counts). 
    Paramter: df
    """
    df_obj = obj_df(df)
    obj_cols = pd.DataFrame(df_obj.dtypes, columns=['dtypes'])
    obj_cols['unique_values'] = df_obj.nunique()
    return obj_cols

# %%
def obj_value_counts(df):
    """
    Returns the counts of unique values in the object columns in the dataframe.
    Parameter: df
    """
    df_obj = obj_df(df)
    for col in df_obj.columns:
        print(df_obj[col].value_counts())
        print('-'*100)

# %%
def num_df(df):
    """
    Returns the dataframe only containing float or int columns
    Paramter: df
    """
    float_mask = np.array(df.dtypes == 'float')
    int_mask = np.array(df.dtypes == 'int')
    mask = float_mask | int_mask
    df_obj = df.iloc[:, mask]
    return df_obj