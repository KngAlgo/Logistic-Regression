from sklearn.linear_model import LogisticRegression

import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_true = pd.read_csv('gender_submission.csv')

x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
print(x_train)


x_test = test_data

model = LogisticRegression()

