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

# standardization
mass_mean = myStat.expectation(X['mass'])
mass_std =  myStat.standard_deviation(X['mass'])
X['mass'] = [
    (mass - mass_mean) / mass_std
    for mass in X['mass']
]

width_mean = myStat.expectation(X['width'])
width_std =  myStat.standard_deviation(X['width'])
X['width'] = [
    (width - width_mean) / width_std
    for width in X['width']
]

height_mean = myStat.expectation(X['height'])
height_std =  myStat.standard_deviation(X['height'])
X['height'] = [
    (height - height_mean) / height_std
    for height in X['height']
]

# Normalization | min-max scale
X_min = min(X['mass'])
X_max = max(X['mass'])
X['mass'] = [
    (mass - X_min) / (X_max - X_min)
    for mass in X['mass']
]

X_min = min(X['width'])
X_max = max(X['width'])
X['width'] = [
    (width - X_min) / (X_max - X_min)
    for width in X['width']
]

X_min = min(X['height'])
X_max = max(X['height'])
X['height'] = [
    (height - X_min) / (X_max - X_min)
    for height in X['height']
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
for i in range(1, 16):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    
    # TODO: Cross Validation 3-8 folds
    cvr = cross_validation_range(knn, X, y, 3, 8) # not trained model!
    accs.append(cvr['accuracy'])
    print('Number Of Folds: ', cvr['numberOfFolds'])
    
    # TODO: LEAVE ONE OUT
    # loo = leave_one_out(knn, X, y)
    # knn = loo['trainedModel']
    # accs.append(loo['accuracy'])
    
    # y_pred = knn.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # accs.append(acc)
    # print(acc)
    
    # TODO: First Task
    # acc = model_train(knn, X_train, X_test, y_train, y_test)['accuracy']
    # accs.append(acc)
    # print(f"Accuracy of {i} = {acc}")

print() # new line

max_acc = max(accs)
print(f"Max accuracy = {max_acc}")
print(f"K = {accs.index(max_acc) + 1}")
print() # new line

# print(accs)