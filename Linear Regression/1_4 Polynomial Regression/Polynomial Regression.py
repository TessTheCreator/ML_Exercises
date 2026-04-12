#Add import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Assign the data to predictor and outcome variables
# Load the data
train_data = pd.read_csv('data.csv')
X = train_data['Var_X']
y = train_data['Var_Y']

# Create polynomial features
# Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(interaction_only=True)
X_poly = poly_feat.fit_transform(X.values.reshape(-1, 1))

# Make and fit the polynomial regression model
# Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression().fit(X_poly,y)

print(f"Coefficients: {poly_model.coef_}")
print(f"Intercept: {poly_model.intercept_}")

# Plot the data and polynomial regression model
X_new = np.linspace(-3, 3.5, 100).reshape(100, 1)
X_new_poly = poly_feat.transform(X_new)
y_new = poly_model.predict(X_new_poly)

plt.scatter(X, y, label="Data")
plt.plot(X_new, y_new, "r-", label="Prediction")
plt.xlabel("Var_X")
plt.ylabel("Var_Y")
plt.legend(loc="best")
plt.show()

