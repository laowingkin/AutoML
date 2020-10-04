# use autokeras to find a model for the sonar dataset
#from numpy import asarray
#from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autosklearn.classification import AutoSklearnClassifier
# load dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
url = 'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'
dataframe = pd.read_csv(url)
print(dataframe.shape)
# split into input and output elements
#data = dataframe.values
data = dataframe.copy()
#X, y = data[:, :-1], data[:, -1]
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
cat_list = ['Pclass', 'Sex', 'SibSp', 'Ticket', 'Cabin', 'Embarked']

for i in X.columns:
    if i in cat_list:
        X[i] = X[i].astype('category')
        X[i] = X[i].cat.add_categories("Missing").fillna("Missing")
    else:
        X[i] = X[i].astype('float')
        X[i].fillna(0, inplace = True)
print(X.isna().sum())
y = data['Survived']
print(X.shape, y.shape)
# basic data preparation
#X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define search
model = AutoSklearnClassifier(time_left_for_this_task=5*60, per_run_time_limit=30)
# perform the search
model.fit(X_train, y_train)

# summarize
print(model.sprint_statistics())
# evaluate best model
y_hat = model.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Accuracy: %.3f" % acc)