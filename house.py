import numpy as np 	
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

data = pd.read_csv(r"C:\Users\SAHAJ\Downloads\House_data.csv")

# Independent variables (remove id, date, price)
X = data.iloc[:, 3:].values   # from bedrooms onward

# Dependent variable (price column)
y = data.iloc[:, 2].values    # price column

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


#bias is based on train data 
#varisance is based on test data

# Check model performance
bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")


train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

y_pred = regressor.predict(X_test)
y_house_price = regressor.predict([X_test[0]])

print(f"house price = {y_house_price[0]:,.2f}")

filename = 'house_predict_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as house_predict_model.pkl")