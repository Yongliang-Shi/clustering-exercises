import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# %%
def kmeans_labels(df_scaled, cols_to_cluster, k): 
    """
    Return in dataframe the labels of each point by KMeans
    Parameter: df_scaled(the scaled dataframe), cols_to_cluster(list of columns to cluster on), k(number of clusters to form)
    The parameters of KMeans are set as default.  
    To make the randomness deterministic, set the random_state=1
    """
    X = df_scaled[cols_to_cluster]
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X)
    labels = pd.DataFrame(kmeans.labels_, columns=['cluster'], index=df_scaled.index)
    return labels

# %%
def viz_kmeans_clustering(df, col_x, col_y):
    """
    Create a scatterplot colorred by cluster with centers shown as 'x'.
    Parameters: df(dataframe with cluster column), col_x(str, column used for x-axis), col_y(str, column used for y-axis)
    """
    fig, ax = plt.subplots(figsize=(13,7))
    for cluster, subset in df.groupby('cluster'):
        ax.scatter(subset[col_x], subset[col_y], label=cluster)
    ax.legend(title='cluster')
    ax.set(ylabel=col_y, xlabel=col_x)
    df.groupby('cluster').mean().plot.scatter(y=col_y, x=col_x, marker='x', s=500, ax=ax, c='black')

# %%
def viz_elbow_method_kmeans(df_scaled, cols_to_cluster, max_k):
    """
    Plot the inertia of kmeans over ks range from 1 to max_k
    Parameters: df_scaled(the scaled dataframe), cols_to_cluster(features to cluster on), max_k(the upper limit of the clusters to form)
    To make the randomness deterministic, set the random_state=1
    """
    X = df_scaled[cols_to_cluster]
    output = {}
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(X)
        output[k] = kmeans.inertia_
    ax = pd.Series(output).plot(figsize=(13,7))
    ax.set(xlabel='k', ylabel='inertia', xticks=range(1, max_k+1), title='The elbow method for determing k')
    ax.grid()