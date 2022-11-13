import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

tips = sb.load_dataset('tips')
print(tips.shape)
print(tips.head())

# viz tip vs total bill
# color the points based on someother value
sb.relplot(data=tips, x='total_bill', y='tip', hue='sex', size='size')
plt.show()

iris = sb.load_dataset('iris')
print(iris.head())

# scatter plot
sb.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species')
plt.show()

titanic = sb.load_dataset('titanic')
print(titanic.head())

# count plot (i.e barchat, but fine to specify only class)
sb.countplot(data=titanic, x='class')
plt.show()

# bar chart
sb.barplot(data=titanic, x='sex', y='survived', hue='class')
plt.show()

# distribution plot
sb.distplot(titanic['age'], kde=True)
plt.show()


# heat map (gives idea about correlation)
corr = titanic.corr()

sb.heatmap(corr, annot=True)
plt.show()