import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import acquire, prepare

# %%
def wrangle_mall_clustering():
    mall = acquire.get_mall_data()
    mall = prepare.encode_label(mall, ['gender'])
    train, validate, test = prepare.split_my_data(mall, pct=0.15)
    train_scaled, validate_scaled, test_scaled = prepare.scale(train, validate, test, ['age', 'annual_income', 'gender_Male'])
    return train_scaled, validate_scaled, test_scaled