# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# Example: Load a CSV file
data = pd.read_csv('/Users/shivenshukla/Desktop/Data Science/Kaggle/input/interest_dataset.csv')

# print(data)

# Assume dataset has columns 'X' (independent variable) and 'Y' (dependent variable)
X = data[['Time (Years)']]  # Features (ensure it's 2D using double brackets)
Y = data['Interest ($)']    # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=32)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the results (for a single feature)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Regression Analysis')
plt.xlabel('Time (Years)')
plt.ylabel('Interest ($)')
plt.legend()
plt.show()
