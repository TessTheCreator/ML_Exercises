#Creating a basic model using multiple linear regressions (where we use multiple variables)

# Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load the data from the boston house-prices dataset
housing_data = fetch_california_housing()
x = housing_data['data']
y = housing_data['target']

# Make and fit the linear regression model
#Fit the model and Assign it to the model variable
model = LinearRegression()
model.fit(x, y)

# Make a prediction using the model
sample_house = [[2.29690000e+00, 1.00000000e+01, 1.05900000e+01, 1.00000000e+00,
                 4.89000000e+02, 6.32600000e+00, 5.25000000e+01, -1.42490000e+02]]

#Predict housing price for the sample_house
prediction = model.predict(sample_house)
print(prediction)