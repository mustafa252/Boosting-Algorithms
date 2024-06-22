
# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# load data 
df = pd.read_csv('heart-disease.csv')
df.columns


############################################################################################
############ data anlaysis

# info
df.info()
# check null values
df.isnull().sum()
# statistical analysis
df.describe()

# target (heartack = 1, non=0)
df['target'].value_counts()



############################################################################################
############ training

x = df.drop('target', axis=1)
y = df['target']


#startified
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

x_train.shape, y_train.shape
x_test.shape, y_test.shape


# check for stratify
y_train.value_counts()
y_test.value_counts() 

############################################################################################
############ training

# ## pip install catboost ..
# import model
from catboost import CatBoostClassifier

# classifier
classifier = CatBoostClassifier()

# fit
classifier.fit(x_train, y_train,)
# predict
y_pred = classifier.predict(x_test)


# classifiacation report
print(classification_report(y_test, y_pred))


############################################################################################
############ hyperparameter tuninig

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# hyperparameters set
params = {'n_estimators': [50, 100, 150, 200, 250],
          'learning_rate':[0.001, 0.01, 0.1, 1, 10],
          'depth': [1,5, 10, 15, 20]}


# Grid Search
grid = GridSearchCV(classifier,
                    param_grid=params,
                    cv=5,
                    n_jobs=-1)

grid.fit(x_train, y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(x_test)

print(classification_report(y_test, y_pred))



# Random Search
grid = RandomizedSearchCV(classifier,
                    params,
                    cv=5,
                    n_jobs=-1)

grid.fit(x_train, y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(x_test)


# classifiacation report
print(classification_report(y_test, y_pred))



































