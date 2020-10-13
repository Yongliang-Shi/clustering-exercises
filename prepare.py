import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

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
def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the series.
    The values returned will be either 0 (if the point is not an outlier), or a number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

# %%
def upper_outlier_columns(df, k):
    '''
    Return the column(s) with the upper_outliers for all the numeric columns in the given dataframe.
    '''
    df1 = pd.DataFrame()
    for col in df.select_dtypes('number'):
        df1[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df1

# %%
def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the series.
    The values returned will be either 0 (if the point is not an outlier), or a number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: min([x - lower_bound, 0]))

# %%
def lower_outlier_columns(df, k):
    '''
    Return the column(s) with the lower_outliers for all the numeric columns in the given dataframe.
    '''
    df1 = pd.DataFrame()
    for col in df.select_dtypes('number'):
        df1[col + '_lower_outliers'] = get_lower_outliers(df[col], k)
    return df1

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