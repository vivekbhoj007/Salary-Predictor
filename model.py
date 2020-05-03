# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:38:52 2020

@author: vbhoj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('salary.csv')
df_actuals=pd.read_csv('salary_new_actuals.csv')

df_new2=pd.concat([df,df_actuals])

# Reading data into X and Y variables
X=df_new2.iloc[:,:-1]
Y=df_new2.iloc[:,1]

'''
#splitting data into training and test sets
# Do we really need to split the data? think :)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
'''
#Implement the reg object based on Simple Linear Regression
#Why are we fiiting and creating the model again, think :)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)

# saving model file into disk
pickle.dump(reg, open('model.pkl','wb'))


#comparing the results
model = pickle.load(open('model.pkl','rb'))
# val = pd.values

print(model.predict([[13]]))

