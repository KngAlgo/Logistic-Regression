from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Importing + Creating Vars for Data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
true_data = pd.read_csv('gender_submission.csv')

y_true = true_data['Survived']

x_train = train_data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y_train = train_data['Survived']

x_test = test_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

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

model = LogisticRegression(max_iter=200)

model.fit(x_train, y_train)

# Create Predictions

y_pred = model.predict(x_test)

plt.figure(figsize=(6,4))
plt.bar(['Not Survived', 'Survived'], [sum(y_pred == 0), sum(y_pred == 1)], color=['red', 'green'])
plt.title('Predicted Survival Counts')
plt.ylabel('Number of Passengers')
plt.xlabel('Survival Prediction')
plt.show()

# 253 TN, 13 FP, 12 FN, 140 TP
print(confusion_matrix(y_true, y_pred))

accuracy = (418-25)/425 #0.9247
print(accuracy)