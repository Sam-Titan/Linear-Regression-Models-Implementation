# Linear Regression on California Housing Prices
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual Results:", y_test)
print("Predictions:", y_pred)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", MSE)
print("Mean Absolute Error:", MAE)
print("R2 score:", R2)

plt.figure(figsize=(8,6)) 

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.3, label='Actual vs Predicted') 

# Plot the "Perfect Prediction" line
# If y_pred == y_test, all points would lie on this line
perfect_line = [y_test.min(), y_test.max()]
plt.plot(perfect_line, perfect_line, color='red', lw=2, linestyle='--', label='Perfect Fit') 

plt.title('Linear Regression: Actual vs Predicted Housing Prices')
plt.xlabel('Actual Prices (Units of $100k)')
plt.ylabel('Predicted Prices (Units of $100k)')

# Add metrics as text
plt.text(0.02, 0.95, # 2% from left, 95% from bottom
         f'MSE: {MSE:.3f}\nMAE: {MAE:.3f}\nR²: {R2:.3f}', 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment='top',   # This ensures the box grows DOWNWARD from the coordinate
         horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.legend()
plt.grid(True)
plt.show()
