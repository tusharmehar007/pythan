# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:49:02 2026

@author: SAHAJ
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data = pd.read_csv(r"C:\Users\SAHAJ\Downloads\emp_sal.csv")


# Independent (X) and dependent (y)
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Import model
from sklearn.linear_model import LinearRegression

# Create model
lin_reg = LinearRegression()

# Train model
lin_reg.fit(X, y)

# Plot graph
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')

plt.title('Linear Regression graph')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5) 
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicton 
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred