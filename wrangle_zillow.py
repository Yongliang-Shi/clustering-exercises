import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import prepare

# %%
def wrangle_zillow_mvp(zillow):
    """
    Wrangle zillow dataset: 
    1. keep the last transaction for each property
    2. keep the single unit properies only (260, 261, 262, 279)
    3. keep the prop_required_column = 0.6 and prop_required_rwo = 0.75
    4. Drop all the remaining_missings
    """
    zillow = prepare.keep_last_transaction(zillow)
    single_unit = [260, 261, 262, 279]
    zillow = zillow[zillow.propertylandusetypeid.isin(single_unit)]
    zillow = prepare.handle_missing_values(zillow, 0.6, 0.75)
    zillow = prepare.drop_remaining_missings(zillow)
    return zillow