# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:11:00 2018

@author: vishawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel("food_truck.xlsx")

X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=0)

from sklearn.linear_model import LinearRegression
simple_liner_reg = LinearRegression()
simple_liner_reg.fit(X_train,y_train)

y_predict = simple_liner_reg.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simple_liner_reg.predict(X_train))

plt.scatter(X_test,y_test,color='blue')

plt.show()