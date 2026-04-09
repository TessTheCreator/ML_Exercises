#strengths and weaknesses of linear regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

# For reproducibility
np.random.seed(0)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()


def ShowRegularRegression():
    lin_reg.fit(X, y)

    plt.scatter(X, y)
    plt.plot(X, lin_reg.predict(X), color='red')
    plt.title("Linear Fit")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()


def ShowLinearRegressionWithOutlier():
    X_out = np.vstack([X, [[2.5]]])
    y_out = np.vstack([y, [[40]]])  # Add a strong outlier

    lin_reg_out = LinearRegression().fit(X_out, y_out)

    plt.scatter(X_out, y_out)
    plt.plot(X_out, lin_reg_out.predict(X_out), color='orange', label="With Outlier")
    plt.plot(X, lin_reg.predict(X), color='red', label="Original")
    plt.legend()
    plt.title("Impact of Outliers")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def ShowPolynomialRegression():
    #(Ridge vs Lasso)
    # You can increase the degree (e.g., degree=9) to observe overfitting behavior

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    lin_poly = LinearRegression().fit(X_poly, y)

    plt.scatter(X, y)
    X_range = np.linspace(0, 2, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    plt.plot(X_range, lin_poly.predict(X_range_poly), color='green')
    plt.title("Polynomial Regression")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def ShowRegularization():
    poly = PolynomialFeatures(degree=9)
    X_poly = poly.fit_transform(X)

    # 1. Linear Regression
    lin_poly = LinearRegression().fit(X_poly, y)

    # 2. Ridge Regression
    ridge_poly = Ridge(alpha=1.0).fit(X_poly, y)

    # 3. Lasso Regression
    lasso_poly = Lasso(alpha=1e-3, max_iter=10000).fit(X_poly, y)

    # Plotting
    X_range = np.linspace(0, 2, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label='Data')

    plt.plot(X_range, lin_poly.predict(X_range_poly), label="Linear (deg=9)", color='green')
    plt.plot(X_range, ridge_poly.predict(X_range_poly), label="Ridge", linestyle='--', color='blue')
    plt.plot(X_range, lasso_poly.predict(X_range_poly), label="Lasso", linestyle='-.', color='red')

    plt.title("Linear Regression (degree=9) with Regularization")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()


ShowRegularRegression()
ShowLinearRegressionWithOutlier()
ShowPolynomialRegression()
ShowRegularization()
