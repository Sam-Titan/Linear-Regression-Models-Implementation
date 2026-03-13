# Here instead of implementing the Linear Regression model without importing any function from sklearn library, I will use the LinearRegression class from sklearn library to fit the model and make predictions.

from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
# Ideal Dataset with no outliers and perfect linear relationship between X and Y
# Study_Hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Scores = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# Dataset with some noise and outliers
Study_Hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Scores = [18, 42, 58, 77, 95, 123, 135, 162, 175, 198]

X = np.array(Study_Hours).reshape(-1, 1)
y = np.array(Scores)

model = LinearRegression()
model.fit(X, y)

predicted_scores = model.predict(X)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted Scores:", predicted_scores)

# Adding Mean Squared Error to evaluate the model
MSE = mean_squared_error(y, predicted_scores)
print("Mean Squared Error:", MSE)