# linear-regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data from CSV
data = pd.read_csv('/Users/prabh/Downloads/Healthcare-Diabetes.csv')

# Select the independent variables
X = data[['Glucose', 'Insulin', 'BMI', 'Age', 'SkinThickness', 'BloodPressure']]
y = data['Outcome']

# Create a LinearRegression model
model = LinearRegression()
model.fit(X, y)

# Predicted values
y_pred = model.predict(X)

# Plot the data points and the multiple linear regression line
plt.scatter(X['Glucose'], y, color='black', label='Glucose Data Points')
plt.scatter(X['Insulin'], y, color='blue', label='Insulin Data Points')
plt.scatter(X['BMI'], y, color='green', label='BMI Data Points')
plt.scatter(X['Age'], y, color='orange', label='Age Data Points')
plt.scatter(X['SkinThickness'], y, color='purple', label='Skin Thickness Data Points')
plt.scatter(X['BloodPressure'], y, color='pink', label='Blood Pressure Data Points')

# Combine all independent variables for the regression line
X_combined = X.sum(axis=1)

# Sort the X_combined and y_pred for a smooth line
sorted_indices = np.argsort(X_combined)
X_combined_sorted = X_combined.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.plot(X_combined_sorted, y_pred_sorted, color='red', label='Multiple Linear Regression Line')

plt.xlabel('Independent Variables')
plt.ylabel('Outcome (y)')
plt.title('Multiple Linear Regression for Diabetes')
plt.legend()
plt.show()
