# Linear Regression with Train-Test Split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

np.random.seed(42)

X = np.random.rand(50, 1) * 100  

Y = 3.5 * X + np.random.randn(50, 1) * 20 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
Slope = model.coef_[0][0]
Intercept = model.intercept_[0]

# Printing the slope and intercept of the regression line
print("Slope (Coefficient):", Slope)
print("Intercept:", Intercept)

MSE = mean_squared_error(Y_test, y_pred)
MAE = mean_absolute_error(Y_test, y_pred)
RMSE = root_mean_squared_error(Y_test, y_pred)
R2 = r2_score(Y_test, y_pred)

n = len(Y_test)
k = 1  # number of independent variables
Adjusted_R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)
print("Mean Squared Error:", MSE)
print("Mean Absolute Error:", MAE)
print("Root Mean Squared Error:", RMSE)
print("Coefficient of Determination (R^2):", R2)
print("Adjusted R squared Error:", Adjusted_R2)

plt.figure(figsize=(8,6)) 
plt.scatter(X_test, Y_test, color='blue', label='Data Points') 
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset using Train-Test Split')
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
