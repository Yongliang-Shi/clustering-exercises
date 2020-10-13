import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import acquire

# %%
def keep_last_transaction(zillow):
    zillow.drop_duplicates(subset=['parcelid'], keep='last', inplace=True, ignore_index=True)
    return zillow

# %%
def handle_missing_values(df, prop_required_column, prop_required_row):
    """
    Drop rows and columsn based on the perent of values that are missing.
    Parameters: 
    1. df
    2. the proportion, for each column, of rows with non-missing values requied to keep the column
    3. the proportion, for each row, of columns with non-missing values required to keep the row
    """
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# %%
def drop_remaining_missings(df):
    """
    Drop all the reamining missing values
    Parameter: df
    """
    mask = (df.isna().sum(axis=1) == 0)
    df = df[mask]
    return df

# %%
def scale(train, validate, test, columns_to_scale):
    """
    Scale the columns using MinMaxScaler
    Parameters: train(df), validate(df), test(df), columns_to_scale(list)
    """
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test