import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# For reproducibility
np.random.seed(0)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=2.0, random_state=0)

X_nl, y_nl = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 class_sep=0.5, flip_y=0.1, random_state=1)

y = 2 * y - 1  # convert labels from {0,1} to {-1,1}

def PerceptionOnSeperableData(y):
    y = 2 * y - 1  # convert labels from {0,1} to {-1,1}

    model = Perceptron(max_iter=1000)
    model.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create decision boundary
    xx = np.linspace(xlim[0], xlim[1])
    w = model.coef_[0]
    b = model.intercept_[0]
    yy = -(w[0] * xx + b) / w[1]

    plt.plot(xx, yy, 'k--')
    plt.title("Perceptron on Linearly Separable Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def PerceptionOnUnseperableData(X_nl, y_nl):

    y_nl = 2 * y_nl - 1

    model_nl = Perceptron(max_iter=1000)
    model_nl.fit(X_nl, y_nl)

    plt.scatter(X_nl[:, 0], X_nl[:, 1], c=y_nl, cmap='bwr', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1])
    w = model_nl.coef_[0]
    b = model_nl.intercept_[0]
    yy = -(w[0] * xx + b) / w[1]

    plt.plot(xx, yy, 'k--')
    plt.title("Perceptron on Non-Linearly Separable Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


PerceptionOnSeperableData(y)
PerceptionOnUnseperableData(X_nl, y_nl)

X_train, X_test, y_train, y_test = train_test_split(X_nl, y_nl, test_size=0.2, random_state=42)

model_eval = Perceptron(max_iter=1000)
model_eval.fit(X_train, y_train)

y_pred = model_eval.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
