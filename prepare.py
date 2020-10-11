import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import acquire


def clean_zillow(zillow):
    zillow.drop_duplicates(subset=['parcelid'], keep='last', inplace=True, ignore_index=True)
    return zillow

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