import pandas as pd
import os
import env

#%%
def get_connection(db, user=env.user, host=env.host, password=env.password):
    """
    Returns the connection string to Codeup Database.
    Parameter: db name (str)
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#%%
def get_zillow_data(query, iteration):
    """
    Returns in df format zillow dataset according to the querry.
    The df has the suffix indicating the version number.
    Parameters: querry(str), iteration(int)
    """
    zillow_csv = 'zillow_' + iteration + '.csv'
    filename = zillow_csv
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        df = pd.read_sql(query, get_connection('zillow'))
        df.to_csv(filename)
        return df

# %%
def get_mall_data():
    """
    Returns mall dataset in df.
    """ 
    filename = 'mall_customers.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql("""select * from customers""", get_connection('mall_customers'))
        df.to_csv(filename)
        return df