def cross_validation_range(model, X, y, i, j):
    accs = []
    for j in range(i, j+1):
        results = cross_validate(estimator=model, X=X, y=y, cv=j)
        accs.append( myStat.expectation(results['test_score']) )
        
    return {
        'trainedModel': model,
        'accuracy': max(accs),
        'numberOfFolds': accs.index(max(accs))
    }

def leave_one_out(model, X, y):
    loo = LeaveOneOut()
    
    accs = []
    for train_index, test_index in loo.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        
        accs.append(model_train(model, X_train, X_test, y_train, y_test)['accuracy'])
        
    return {
        'trainedModel': model,
        'accuracy': myStat.expectation(accs)
    }

def model_train(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    return {
        'trainedModel': model,
        'accuracy': accuracy_score(y_test, y_pred)
    }

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:18:55 2022

@author: abd_i
"""
#import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statistics
import math


#get DataSet
fruits = pd.read_csv("H:\\Education\College\\04 Final year\\AI 2\\ML\\Lab 02\\Python File\\fruit_data_with_colours.csv")

fruits.head()

Fruit_name =  dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique() ))

print(type(Fruit_name))
print(Fruit_name)

#get independent_variable (Features)
X = fruits[['mass','width','height']]
X.head(10)

#get dependent_variable (Response)
y= fruits['fruit_label']
y.head(10)

# TODO: preprocessing (applying normal distribution)
import myStatistics as myStat

for feature in X:
    # standardization | Z-scale
    mean = myStat.expectation(X[feature])
    std =  myStat.standard_deviation(X[feature])
    X[feature] = [
        (data - mean) / std
        for data in X[feature]
    ]
    
    # Normalization | min-max scale
    X_min = min(X[feature])
    X_max = max(X[feature])
    X[feature] = [
        (data - X_min) / (X_max - X_min)
        for data in X[feature]
    ]

#Split data 
X_train,X_test,y_train,y_test =  train_test_split(X,y,random_state=42)


#Classifier using KNN
from sklearn.neighbors import KNeighborsClassifier

### added
# accuracy method of KNN
# take (test_label, predicted_values)
from sklearn.metrics import accuracy_score

# cross validation
from sklearn.model_selection import cross_validate

# leave one out
from sklearn.model_selection import LeaveOneOut

### added
accs = []
folds = []
for i in range(1, 16):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    
    # TODO: Cross Validation 3-8 folds
    cvr = cross_validation_range(knn, X, y, 3, 8) # not trained model! 3->8
    accs.append(cvr['accuracy'])
    folds.append(cvr['numberOfFolds'])
    print('KNN k: ', i)
    print('No. Folds That Give Max Accuracy: ', cvr['numberOfFolds'])
    print('Accuracy: ', cvr['accuracy'])
    
    # TODO: LEAVE ONE OUT
    # loo = leave_one_out(knn, X, y)
    # knn = loo['trainedModel']
    
    #! Run the first two line With 1) or 2) Not with both in the same time to get correct values
    
    #! 1)
    # accs.append(loo['accuracy'])
    # print(loo['accuracy'])
    
    #! 2)
    # y_pred = knn.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # accs.append(acc)
    # print(acc)
    
    # TODO: First Task
    # acc = model_train(knn, X_train, X_test, y_train, y_test)['accuracy']
    # accs.append(acc)
    # print(f"Accuracy of {i} = {acc}")

print()
print('=======================================================')
print()

max_acc = max(accs)
print(f"Max accuracy = {round(max_acc * 100, 2)}%")
print(f"K of KNN = {accs.index(max_acc) + 1}")
if 'cvr' in vars():
    print('No. Folds: ', folds[accs.index(max_acc)])
print() # new line

# print(accs)