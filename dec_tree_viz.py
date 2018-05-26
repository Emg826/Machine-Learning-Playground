
"""
Created on Wed May 23 20:57:05 2018 on my computer

Decision tree viz

Observation: spectral clustering is much better at DBSCAN when there
is not much to tell the data apart. See the diagram at:
http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

Note: PCA upped it to 27% and accuracy from 70% to 79%
ACTUALLY... i didn't even have a clustering method in code, os just on consodle,
the performance is poor

@author: EricG
"""
from sklearn.decomposition import PCA
import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

filepath = '/Users/EricG/Downloads/car.data.txt'
df = pandas.read_csv(filepath)  
df.columns = ['buying',
              'maint',
              'doors',
              'persons',
              'lug_boot',
              'safety',
              'class',
              ]
# convert nearly quantitative data to completely quant
# Benefit: fewer dimensions in the data as if dummy encoded; woohoo!
# convert '5more' to 7; lambda's w/ if's also must have else's
df.doors = df.doors.apply(lambda x: 7 if x == '5more' else x)

# same idea, but now w/ persons
df.persons = df.doors.apply(lambda x: 7 if x == 'more' else x)

not_need_dummy = ['class', 'doors', 'persons']
need_dummy_encode = df.drop(not_need_dummy, axis=1).columns # all input cols have a categorical var
not_need_dummy.pop(0)
                           
need_label_encode = 'class' # don't need dummy since it is a target variable?

# encode targets
le = LabelEncoder()
y = le.fit_transform(df[need_label_encode]).astype(np.int)

# convert categorical to binary data by adding col's for each categorical feature
x = pandas.get_dummies(df[need_dummy_encode]).join(df[not_need_dummy])

# Do some PCA stuff before clustering or classifying
col_names = x.columns # col names are obliterated when fit_transform, so save for after
x = pandas.DataFrame(PCA().fit_transform(x.values, y))
x.columns = col_names

y = list(y)

from sklearn.tree import DecisionTreeClassifier, export_graphviz
decision_tree = DecisionTreeClassifier(max_depth=5, criterion='gini',
                                       min_samples_split=25, 
                                       min_weight_fraction_leaf=0.02)

decision_tree = decision_tree.fit(x.values, y)
y_predict = decision_tree.predict(x.values, y)

# Print and output some results of the decisontree
from sklearn.metrics import accuracy_score
print(accuracy_score(y, y_predict))
var_names = pandas.get_dummies(df[need_dummy_encode]).columns

export_graphviz(decision_tree, out_file='tree.dot', feature_names = var_names,
                filled=True, impurity=False, proportion=True, rounded=True,
                leaves_parallel=False,)

import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')



