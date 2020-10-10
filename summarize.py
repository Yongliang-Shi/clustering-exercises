def object_cols (df):
    """
    Returns the object columns of the dataframe and their unique values (don't include NaN in counts). 
    Paramters: df
    """
    mask = np.array(df.dtypes == 'object')
    obj_df = df.iloc[:, mask]
    obj_cols = pd.DataFrame(obj_df.dtypes, columns=['dtypes'])
    obj_cols['unique_values'] = obj_df.nunique()
    return obj_cols