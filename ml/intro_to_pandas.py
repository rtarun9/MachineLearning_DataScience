import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)

# finding number of rows and columns of df
print(df.shape)

# print first 5 rows in data frame
print(df.head())

# print last 5 rows in data frame
print(df.tail())

# information about data type (how many things are null and how many are not)
print(df.info())

# Find number of missing values in each column
print(df.isnull().sum())

# print number of values based on label (I.E header of column)
df2 = pd.read_csv('data_sets/nba_salaries.csv')
print(df2.head())

# group all the teams and display the count of each.
print(df2.value_counts('TEAM'))

# alternate method : group by based on mean
print(df2.groupby('TEAM').mean())

print(df2.columns)
print(df2.groupby('TEAM')['SALARY'].sum())


# statistical measures

# column wise data
print('mean : ', df.mean())
print('standard deviation : ', df.std())
print('min : ', df.min())
print('max : ', df.max())
print('mode : ', df.mode())

# all statistical measures about the df
print(df.describe())

# correlation
print(df.corr())

# removing row
new_df = df.drop(index=0, axis=0)
print(df.shape, new_df.shape)

# add column to df
new_df['XYZ'] = new_df['HouseAge']
print(new_df.shape)

# locating data using iloc
print('second row : ', df.iloc[2])
print('first 5 rows and first 5 columns : ', df.iloc[0:5, 0:5])