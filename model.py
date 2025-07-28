from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

# Importing + Creating Vars for Data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_true = pd.read_csv('gender_submission.csv')

x_train = train_data.drop(columns=['Survived', 'Name', 'Ticket'], axis=1)
y_train = train_data['Survived']

x_test = test_data.drop(columns=['Name', 'Ticket'], axis=1)

# Scaling

scaler = StandardScaler()

x_train['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})
x_train['Embarked'] = x_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
x_test['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})

# Model Instantiation and Fitting

model = LogisticRegression()

model.fit(x_train, y_train)

