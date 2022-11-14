import numpy as np
from sklearn import linear_model


x = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(x, y)

# predict (should be yes)
predicted = logr.predict(np.array([5.46]).reshape(-1, 1))
print(predicted)

# using the credit card fraud detection dataset.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_sets/creditcard.csv')
df.info()

# drop duplicate values
df.drop_duplicates(inplace=True)

# drop the time column
df.drop('Time', axis=1, inplace=True)
x = df.iloc[:,df.columns != 'Class']
y = df.Class
df.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

train_acc = model.score(x_train_scaled, y_train)
print("Accuracy for training set : ", train_acc * 100)

y_pred = model.predict(x_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print("Accuracy for test set : ", test_acc * 100)

print(classification_report(y_test, y_pred))

c_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 6))
plt.title('confusion matrix')
sb.heatmap(c_mat, annot=True)
plt.ylabel('actual values')
plt.xlabel('predicted values')
plt.show()