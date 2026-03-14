# Here instead of implementing the Linear Regression model without importing any function from sklearn library, I will use the LinearRegression class from sklearn library to fit the model and make predictions.

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
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
Slope = model.coef_[0]
Intercept = model.intercept_
print("Slope (Coefficient):", Slope)
print("Intercept:", Intercept)
print("Predicted Scores:", predicted_scores)

# Adding Mean Squared Error to evaluate the model
MSE = mean_squared_error(y, predicted_scores)
print("Mean Squared Error:", MSE)

# Adding Mean Absolute Error to evaluate the model
MAE = mean_absolute_error(y, predicted_scores)
print("Mean Absolute Error:", MAE)

# Adding Root Mean Squared Error, Coefficient of Determination, and Adjusted R squared Error to evaluate the model

# Root Mean Squared Error
RMSE = root_mean_squared_error(y, predicted_scores)
print("Root Mean Squared Error:", RMSE)

# Coefficient of Determination
R2 = r2_score(y, predicted_scores)
print("Coefficient of Determination (R^2):", R2)

# Adjusted R squared Error
n = len(y)
k = 1  # number of independent variables

Adjusted_R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)
print("Adjusted R squared Error:", Adjusted_R2)


plt.figure(figsize=(8,6)) 
plt.scatter(X, y, color='blue', label='Data Points') 
plt.plot(X, predicted_scores, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Dataset with Noise and Outliers using imported functions')
plt.xlabel('X')
plt.ylabel('Y')
# Add metrics as text on the graph
plt.text(0.05, 0.95, 
         f'Slope: {Slope:.3f}\nIntercept: {Intercept:.3f}\nMSE: {MSE:.3f}\nMAE: {MAE:.3f}\nRMSE: {RMSE:.3f}\nR²: {R2:.3f}\nAdjusted R²: {Adjusted_R2:.3f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, 
         verticalalignment='top', 
         bbox=dict(facecolor='white', alpha=0.6))
plt.legend()
plt.grid(True)
plt.show()
