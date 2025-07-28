from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

# Importing + Creating Vars for Data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_true = pd.read_csv('gender_submission.csv')

x_train = train_data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_data['Survived']

x_test = test_data.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)

# Scaling


x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_train['Fare'].fillna(x_train['Fare'].mean(), inplace=True)

x_train = x_train.dropna(subset=['Embarked'])
y_train = y_train.loc[x_train.index]

x_train['Sex'] = x_train['Sex'].map({'male': 0, 'female': 1})
x_train['Embarked'] = x_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

x_test['Embarked'] = x_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
x_test['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})

# Model Instantiation and Fitting

print(x_train.head())

print(x_test.head())
# Show columns with missing values in train_data
print("Missing values in train_data:")
print(x_train.isnull().sum())

# Show columns with missing values in test_data
print("\nMissing values in test_data:")
print(x_test.isnull().sum())

model = LogisticRegression(max_iter=200)

model.fit(x_train, y_train)

