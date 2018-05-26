"""
Created on Thu May 24 19:42:12 2018 on my computer

Testing out some clustering algorithms w/ some actual numerical data
(unlike the dec_tree_viz.py's car data)

Not concerned so much about predicting occupancy as I am about 
just getting familiar w/ the clustering algorithms and how they work.

Specifically: DBSCAN, SpectralClustering, AgglomerativeClustering
Question: should i be doing clustering only on 2 variables and plt for those 2
at a time, or should i be doing 1 clustering that predicts for all pairs
Recomendation: sort df_train and df_test columns so that inputs are in 
alphaebtical order (make figure easier to follow)

https://stackoverflow.com/questions/209840/map-two-lists-into-a-dictionary-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
@author: EricG
"""
import pandas
import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import Birch, DBSCAN, KMeans
from sklearn.cluster import  MiniBatchKMeans, MeanShift, SpectralClustering
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

filepath = '/Users/EricG/Downloads/occupancy_data/datatraining.txt'
df_train = pandas.read_csv(filepath).drop('date', axis=1) # only ready quant data

filepath = '/Users/EricG/Downloads/occupancy_data/datatest.txt'
df_test = pandas.read_csv(filepath).drop('date', axis=1)

# get training and test data into a usable format
x_train = df_train.drop('Occupancy', axis=1).values
x_test = df_test.drop('Occupancy', axis=1).values
                     
y_train = df_train['Occupancy'].values
y_test = df_test['Occupancy'].values


def plot_clusterings(x_actual, y_actual, cluster_object, col_names):
    """
    Plot the results of the clustering algorithm in a single figure
    with many subplots. Colored by predicted cluster, not actual
    
    :param x_actual: the actual input data
    :param y_actual: the actual target data; not the predictions
    :cluster_object: a cluster object
    :col_names: list of names of the columns/variables - 
                assumes that this list is in same order as cols of x_actual
    """
    predictions = list(cluster_object.fit_predict(x_actual, y_actual))
    
    cols_to_plot = col_names[:]
    fig = plt.figure()
    colors = pandas.Series(['red', 'orange', 'yellow',
                            'green', 'blue', 'violet',
                            'darkred', 'darkorange', 'gold',
                            'darkgreen', 'darkblue', 'darkviolet',
                            'black', 'brown', 'gray', 
                            'lightgray', 'brown', 'wheat',
                            'khaki', 'cyan', 'whitesmoke', 
                            'lime', 'crimson', 'cyan',
                            ])
    
    # provide support for -1's in DBSCAN by shifting 'red' down to index=-1, etc.
    colors = colors.reindex([x for x in range(-1, len(colors))]).shift(-1)
    
    # calc number of combinations of columns
    total_num_sub_plots = 0
    for i in range(x_actual.shape[1]): # shape[1] is # cols in 2D matrix or np array
        total_num_sub_plots += i
        
    # Approximate the number of rows and cols for a 3:4 (or is it 4:3?) screen   
    # since most screens are close to 9:16, do something similiar for rows,cols   
    # but use 3:4 instead of 9:16
    # use round(sqrt()) instead of ceil(len / 3 or 4 ) 
    import math
    
    num_rows = round(math.sqrt(total_num_sub_plots))
    num_columns = round(math.sqrt(total_num_sub_plots)) + 1
    
    # Plot all combinations of the columns and their clusters w/ their predictions
    # denoted by color in graph
    num_plotted_subplots = 0           
    
    # necessary since popping from cols_to_plot in nested for's means y_col_idx
    # is pretty much useless
    col_name_idx_dict = dict(zip(cols_to_plot, 
                                 [col_idx for col_idx in range(len(cols_to_plot))])) 
    
    print(col_name_idx_dict)
    for x_col_idx, column_x in enumerate(cols_to_plot):
        for y_col_idx, column_y in enumerate(cols_to_plot[x_col_idx:]):
            if column_x == column_y: # for when get to end & just in case i messed up
                continue
            
            #print(column_x, column_y) uncomment for debug which col's are graphed, when
            num_plotted_subplots += 1
            fig.add_subplot(num_rows, num_columns, num_plotted_subplots)
            
            plt.scatter(x_actual[:, x_col_idx],
                        x_actual[:, col_name_idx_dict[column_y]],
                        color=colors[predictions],
                        s=1)
            plt.xlabel(column_x)
            plt.ylabel(column_y)
    plt.suptitle(cluster_object.__repr__(), fontsize=8)
    plt.show()

            
                  
train_cols = list(df_train.columns)
train_cols.pop()

"""AffinityPropagation 
- by far the slowest; didn't even bother waiting for it to finish"""
#aff = AffinityPropagation(max_iter=20)
#plot_clusterings(x_train, y_train, aff, train_cols)
#print(accuracy_score(y_test, aff.fit_predict(x_test)))

""" Agglomerative Clustering """
print('Agglomerative Clustering')
agglo = AgglomerativeClustering(n_clusters=2)
plot_clusterings(x_train, y_train, agglo, train_cols)
print(accuracy_score(y_test, agglo.fit_predict(x_test)))

"""Birch Clustering"""
print('Birch Clustering')
birch = Birch(threshold=0.4, branching_factor=50, n_clusters=2)
plot_clusterings(x_train, y_train, birch, train_cols)
print(accuracy_score(y_test, birch.fit_predict(x_test)))

"""Feature Agglomeration
- uses fit_transform instead of fit_predict, so can't use w/ graph"""

""" DBSCAN """
print('DBSCAN')
dbscan = DBSCAN(eps=4, min_samples=30)
plot_clusterings(x_train, y_train, dbscan, train_cols)
print(accuracy_score(y_test, dbscan.fit_predict(x_test)))

"""KMeans"""
print('KMeans')
km = KMeans(n_clusters=2)
plot_clusterings(x_train, y_train, km, train_cols)
print(accuracy_score(y_test, km.fit_predict(x_test)))

"""MiniBatchKMeans"""
print('MiniBatchKMeans')
mbkm = MiniBatchKMeans(n_clusters=2, max_iter=150, n_init=3)
plot_clusterings(x_train, y_train, mbkm, train_cols)
print(accuracy_score(y_test, mbkm.fit_predict(x_test)))

"""MeanShift"""
print('MeanShift')
ms = MeanShift()
plot_clusterings(x_train, y_train, ms, train_cols)
print(accuracy_score(y_test, ms.fit_predict(x_test)))

"""Spectral Clustering
-took too long, but it was decent in another scenario, so keeping it"""
print('Spectral Clustering')
#sc = SpectralClustering(n_clusters=2, n_jobs=-1)
#plot_clusterings(x_train, y_train, sc, train_cols)
#print(accuracy_score(y_test, sc.fit_predict(x_test)))

