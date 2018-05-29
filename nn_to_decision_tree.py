"""
Basic idea: train a decision tree on the same inputs as a neural network, but
instead of using the actual data targets, use the mlp's train/test predictions
as targets.

Conclusion: this does not work. Although the decision tree approximation of the
is slightly different (±0.02 generally) than the regular decision tree, it still
is nowhere near as accurate as the nn. Therefore, it is not a good 
approximation of the nn.

Created on Mon May 28 20:13:28 2018

@author: EricG
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

bundle_of_data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(bundle_of_data.data,
                                                    bundle_of_data.target,
                                                    test_size=0.30)

mlp = MLPClassifier(hidden_layer_sizes=(180,180),
                    activation='relu',
                    solver='adam',
                    alpha=0.0005,
                    learning_rate='constant',
                    max_iter=10000,
                    shuffle=False,
                    tol=1e-30,
                    learning_rate_init=0.0001,
                    validation_fraction=0.1,
                    )

mlp.fit(x_train, y_train)

print('RESULTS')
mlp_train_pred = mlp.predict(x_train)
print('mlp actual train preds acc: ', accuracy_score(y_train, mlp_train_pred))

mlp_test_pred = mlp.predict(x_test)
print('mlp actual test preds acc: ', accuracy_score(y_test, mlp_test_pred))

"""
Hypothesis: 
Using the predictions of a neural network as the targets (but w/ same inputs,
e.g., x_train and x_test) should allow the decision tree to approximate
the neural network's rules. If the accuracies on predicting the nn's output are
fairly high and if accuracies of the decision tree for predicting the actual 
targets (y_train and y_test) are the same the accuracies of the mlp,
then the decision tree is an  approximation of the mlp rules?
"""

dt_appx = DecisionTreeClassifier(splitter='best')

dt_appx.fit(x_train, mlp_train_pred)

# results of approximating mlp
dt_appx_mlp_train_pred = dt_appx.predict(x_train)
print()
print('dt_appx mlp train preds acc: ', accuracy_score(mlp_train_pred, 
                                                      dt_appx_mlp_train_pred))
dt_appx_mlp_test_pred = dt_appx.predict(x_test)
print('dt_appx mlp test preds acc: ', accuracy_score(mlp_test_pred, 
                                                     dt_appx_mlp_test_pred))

# results of generalizability (to compare to mlp)
dt_appx_train_pred = dt_appx.predict(x_train)
print('dt_appx actual train preds acc: ', accuracy_score(y_train, 
                                                         dt_appx_train_pred))

dt_appx_test_pred = dt_appx.predict(x_test)
print('dt_appx actual test preds acc: ', accuracy_score(y_test, 
                                                        dt_appx_test_pred))

# Just to compare, let's see a decision tree trained like as a decision tree
# normally is trained.
# At the very least, this may be evidence that training a decision tree on the
# outputs of a neural network can improve its performance if regular_dt's 
# accuracy is lower
dt_regular = DecisionTreeClassifier(splitter='best')
dt_regular.fit(x_train, y_train)

# only train test on actual data since not trying to approximate mlp here
dt_regular_train_pred = dt_regular.predict(x_train)
print()
print('dt_regular actual train preds acc: ', accuracy_score(y_train,
                                                            dt_regular_train_pred))
dt_regular_test_pred = dt_regular.predict(x_test)
print('dt_regular actual test preds acc: ', accuracy_score(y_test,
                                                           dt_regular_test_pred))
"""
Output:
RESULTS
mlp actual train preds acc:  0.934673366834
mlp actual test preds acc:  0.941520467836

dt_appx mlp train preds acc:  1.0
dt_appx mlp test preds acc:  0.964912280702
dt_appx actual train preds acc:  0.934673366834
dt_appx actual test preds acc:  0.929824561404

dt_regular actual train preds acc:  1.0
dt_regular actual test preds acc:  0.947368421053

Conclusion: the accuracy of dt_mlp on predicting mlp's predictions is 
relatively high, and in terms of generalizability, the accuracy on the regular
train and test is almost identical to that of the mlp (or better in some 
instances). Interesting. 

Also, dt_mlp train and ml train acc's are literally identical! 
This occurred on multiple runs.

Also interesting is the fact that the decision tree trained on the mlp preds
(dt_mlp) generalizes more like how mlp generalizes than how dt_regular 
generalizes (I am sort of equating generalizability to test accuracy).

It is hard to distinguish between the decision tree learning the neural 
network and the decision tree learning the data through the neural network.

Big finding is that when 'dt_appx mlp train/test preds acc' is 1.0, then 
the 'Dt actual train/test acc' will be the same as that of
'Mlp train acc'. This, to me, indicates that the decision tree is approximating 
the Mlp. Here is one such run where this happened:
    
RESULTS
mlp actual train preds acc:  0.380952380952
mlp actual test preds acc:  0.222222222222

dt_appx mlp train preds acc:  1.0
dt_appx mlp test preds acc:  1.0
dt_appx actual train preds acc:  0.380952380952
dt_appx actual test preds acc:  0.222222222222

dt_regular actual train preds acc:  0.67619047619
dt_regular actual test preds acc:  0.644444444444

.

One of the problems with this is that there is no telling if the dt_mlp can 
hang with the mlp when dt_regular is significantly worse than the mlp. In other
words, can training a decision tree on the predictions of the mlp result
in a better decision tree than a decision tree trained on the actual targets?
This sounds absurd, but who the heck knows.

Here are some results that might indicate that training on the mlp can in 
some situations might improve the decision tree (in terms of overfitting at
least):
RESULTS
mlp actual train preds acc:  0.91959798995
mlp actual test preds acc:  0.912280701754

dt_appx mlp train preds acc:  1.0
dt_appx mlp test preds acc:  0.929824561404
dt_appx actual train preds acc:  0.91959798995
dt_appx actual test preds acc:  0.912280701754

dt_regular actual train preds acc:  1.0
dt_regular actual test preds acc:  0.894736842105


Actually, let's test this more to see. Let's intentionally make dt_regular and
dt_mlp terrible by setting max_depth to 2. At this point, I think it would
be safe to say that dt_mlp is no longer a good approximation of mlp; however,
it should still be an approximation of mlp. 

Output:

RESULTS
mlp actual train preds acc:  1.0
mlp actual test preds acc:  0.977777777778

dt_appx mlp train preds acc:  0.72380952381
dt_appx mlp test preds acc:  0.555555555556
dt_appx actual train preds acc:  0.72380952381
dt_appx actual test preds acc:  0.533333333333

dt_regular actual train preds acc:  0.72380952381
dt_regular actual test preds acc:  0.533333333333

That was for max_depth=1

For max_depth=2:
RESULTS
mlp actual train preds acc:  0.980952380952
mlp actual test preds acc:  1.0

dt_appx mlp train preds acc:  0.952380952381
dt_appx mlp test preds acc:  0.977777777778
dt_appx actual train preds acc:  0.933333333333
dt_appx actual test preds acc:  0.977777777778

dt_regular actual train preds acc:  0.952380952381
dt_regular actual test preds acc:  0.977777777778

So, it is starting to look like a soft upper bound for dt_appx's test accuracy.
It is related to the best regular decision tree with the 
same max_depth as appx tree.

If this is the case, then why would you ever want to use a decision tree to 
approximate the neural network, right? I mean, the actual test preds acc
of the dt_appx is not as good as the mlp and not better than dt_regular, so 
really ther eis no advantage here. Yet, sometimes I'll get results like
this for max_depth=None and test_size=0.4:
RESULTS
mlp actual train preds acc:  0.980952380952
mlp actual test preds acc:  1.0

dt_appx mlp train preds acc:  1.0
dt_appx mlp test preds acc:  0.955555555556
dt_appx actual train preds acc:  0.980952380952
dt_appx actual test preds acc:  0.955555555556

dt_regular actual train preds acc:  1.0
dt_regular actual test preds acc:  0.933333333333

This is confusing. Perhaps dt_appx could be helpful when dealing with very 
little training data?

Another thing I noticed: dt_appx actual test acc = mlp actual test acc * 
                                                   dt_appx mlp test acc
                                                   
In other words, if dt_appx mlp test preds acc is less than 1, then dt_appx's
dt_appx actual test preds acc will be worse than the nn's. Possibly might
be some sort of an efficiency or goodness-of-approximation measurement, e.g.,
something I could maximize in a for loop?
"""

# Just output the decision tree to a .dot file and then let graphviz convert
# the .dot file to a .png file (an image).
export_graphviz(dt_appx, out_file='dt_appx_of_mlp.dot',
                feature_names=bundle_of_data.feature_names, 
                class_names=bundle_of_data.target_names,
                filled=True, impurity=False, rounded=True,
                proportion=True)

export_graphviz(dt_regular, out_file='dt_regular.dot',
                feature_names=bundle_of_data.feature_names, 
                class_names=bundle_of_data.target_names,
                filled=True, impurity=False, rounded=True,
                proportion=True)

import pydot

(graph,) = pydot.graph_from_dot_file('dt_appx_of_mlp.dot')
graph.write_png('dt_appx_of_mlp.png')

(graph,) = pydot.graph_from_dot_file('dt_regular.dot')
graph.write_png('dt_regular.png')

