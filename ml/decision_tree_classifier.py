import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets 
import seaborn as sb
import graphviz

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
df = pd.read_csv('data_sets/diabetes.csv', header=None, names=col_names)
print(df.head())

df.drop(df.index[0], inplace=True)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
x = df[feature_cols]
y = df.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print('accuracy : ', metrics.accuracy_score(y_test, y_pred))

# Using graph for decision tree (from the iris data set of seaborn)
iris = datasets.load_iris()

x = iris.data
y = iris.target

clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(x, y)


# will work in a jupyter notebook, not in a .py file.
# dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)

# draw graph
# graphviz.Source(dot_data, format='png')
# graphviz.render('tree', view=True)

# Decision tree from scratch

from numpy import log2

outlook ='overcast, overcast, overcast, overcast, rainy, rainy, rainy, rainy, rainy, sunny, sunny, sunny, sunny, sunny'.split(',')
temp='hot, cool, mild, hot, mild, cool, cool, mild, mild, hot, hot, mild, cool, mild'.split(',')
humidity='high, normal, high, normal, high, normal, normal, normal, high, high, high, high, normal, normal'.split(',')
windy='False, True, True, False, False, False, True, False, True, False, True, False, False, True'.split(',')
play='yes, yes, yes, yes, yes, yes, no, yes, no, no, no, no, yes, yes'. split(',')

data_set = {'outlook':outlook, 'temp':temp, 'humidity':humidity, 'windy':windy,'play':play}
df = pd.DataFrame(data_set, columns=['outlook', 'temp', 'humidity', 'windy', 'play'])

print(df)

# Calculate the entropy
entropy_node = 0
values = df.play.unique()

for value in values:
    frac = df.play.value_counts()[value]/len(df.play)
    entropy_node += -frac * np.log2(frac)

print('entropy node', entropy_node)

min_float_value = np.finfo(float).eps

def ent(df, attribute):
    target_variables = df.play.unique()
    variables = df[attribute].unique()
    entropy_attribute = 0
    for variable in variables:
        entropy_each_feat = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df.play==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+min_float_value)
            entropy_each_feat += - fraction*log2(fraction+min_float_value)

        fraction2=den/len(df)
        entropy_attribute += -fraction2*entropy_each_feat
    return(abs(entropy_attribute))

print("Outlook", ent(df, 'outlook'))
print("Humidity", ent(df, 'humidity'))
print("Windy", ent(df, 'windy'))
print("Temperature", ent(df, 'temp'))

# Information gain
def ig(e_dataset, e_attr):
    return(e_dataset - e_attr)

print("Outlook", ig(entropy_node, ent(df, 'outlook')))
print("Humidity", ig(entropy_node, ent(df, 'humidity')))
print("Windy", ig(entropy_node, ent(df, 'windy')))
print("Temperature",ig(entropy_node, ent(df, 'temp')))