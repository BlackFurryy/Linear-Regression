# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 22:24:08 2018

@author: vishawar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

#Loading dataset
dataset = pd.read_excel("housing.xlsx")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,2].values

X_norm_0 = normalize(X[:,0].reshape(-1,1),axis=0,norm='l2')
X_norm = pd.concat([pd.DataFrame(X_norm_0),pd.DataFrame(X[:,1])],axis=1)
X_train_norm,X_test_norm,y_train_norm,y_test_norm = train_test_split(X_norm,y,test_size=.25,random_state=0)

#Splitting train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)

#Applying linear regression on train data set
multi_linear_regression = LinearRegression()
multi_linear_regression_norm = LinearRegression()
multi_linear_regression.fit(X_train,y_train)
multi_linear_regression_norm.fit(X_train_norm,y_train)


#predict on x_test and custom sample
y_predict = multi_linear_regression.predict(X_test)
y_predict_norm = multi_linear_regression_norm.predict(X_test_norm)


#arrr = np.array([1650,3])
#result = multi_linear_regression.predict(arrr.reshape(1,-1))
#arrr_norm = np.array([1650,3])
#arrr_norm = normalize(arrr_norm,norm='l1')
#result_norm = multi_linear_regression_norm.predict(arrr_norm.reshape(1,-1))


squared_sum_predict_error = np.sum(np.square(y_predict-y_test))
squared_sum_predict_norm_error = np.sum(np.square(y_predict_norm-y_test_norm))


#plot
