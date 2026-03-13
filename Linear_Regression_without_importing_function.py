# Here I hope to implemebt Linear Regression model without importing any function from sklearn library. I will use the formula of Linear Regression to calculate the slope and intercept.

# Ideal Dataset with no outliers and perfect linear relationship between X and Y
# Study_Hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Scores = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# Dataset with some noise and outliers
Study_Hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Scores = [18, 42, 58, 77, 95, 123, 135, 162, 175, 198]

# The formula for linear regression is: y = mx + c
# Here m is the slope and c is the intercept. We can calculate m and c using the following formulas:
# m = sum((mean of x) - xI) * (mean of y - yI) / sum((mean of x) - xI)^2
# c = mean of y - m * mean of x
mean_x = sum(Study_Hours) / len(Study_Hours)
mean_y = sum(Scores) / len(Scores)
sum_XY = 0
sum_X_squared = 0
for i in range(len(Study_Hours)):
    sum_XY += (Study_Hours[i] - mean_x) * (Scores[i] - mean_y)
    sum_X_squared += (Study_Hours[i] - mean_x) ** 2

m = sum_XY // sum_X_squared
c = mean_y - m * mean_x
print("Slope (Coefficient):", m)
print("Intercept:", c)

# Now we can use the slope and intercept to predict the scores for the given study hours.
predicted_scores = []

for i in range(len(Study_Hours)):
    score = m*Study_Hours[i] + c
    predicted_scores.append(score)

print("Predicted Scores:", predicted_scores)
print("Comparing with the actual scores:", Scores)

# Adding Mean Squared Error to evaluate the model
# Formula for Mean Squared Error: MSE = sum((yI - y_predicted)^2) / n
Sum_of_squared_residuals = 0
for i in range(len(Study_Hours)):
    Sum_of_squared_residuals += (Scores[i] - predicted_scores[i])**2

MSE = Sum_of_squared_residuals / len(Study_Hours)
print("Mean Squared Error:", MSE)

# Addinf Mean Absolute Error to evaluate the model
# Formula for Mean Absolute Error: MAE = sum(|yI - y_predicted|) / n
Sum_of_absolute_residuals = 0
for i in range(len(Study_Hours)):
    Sum_of_absolute_residuals += abs(Scores[i] - predicted_scores[i])

MAE = Sum_of_absolute_residuals / len(Study_Hours)
print("Mean Absolute Error:", MAE)