import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r"C:\Users\SAHAJ\Downloads\Investment.csv")

X = data.iloc[:, :-1]
Y = data.iloc[:, 4]

X = pd.get_dummies(X,dtype = int)

from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)


X = np.append(arr=np.full((50,1), 42467).astype(int), values=X, axis=1)


# WE USE SKLEARN FORM MAKING MODEL AND STATSMODELS.API LIBRARY IS USE TO TEST ACCURACY OF MEDOL

# OLS ordinary list square method we use and this is in statsmodel.api library for checking accuracy

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test, y_test)
variance