import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# For reproducibility
np.random.seed(0)

def plot_decision_boundary(model, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFBBBB', '#BBBBFF']), alpha=0.4)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# Generate data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=2.0, random_state=0)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Plot decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(model, X, y, ax, "Decision Tree on Linearly Separable Data")
plt.show()

# Plot tree structure
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=["Feature 1", "Feature 2"])
plt.title("Tree Structure")
plt.show()

# Generate complex dataset
X_complex, y_complex = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                           n_informative=2, n_clusters_per_class=1,
                                           class_sep=0.5, flip_y=0.3, random_state=1)

# Train a decision tree without max_depth limitation
model_complex = DecisionTreeClassifier(max_depth=None)
model_complex.fit(X_complex, y_complex)

# Plot decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(model_complex, X_complex, y_complex, ax, "Decision Tree on Complex Data")
plt.show()

# Visualize tree structure
plt.figure(figsize=(12, 6))
plot_tree(model_complex, filled=True, feature_names=["Feature 1", "Feature 2"])
plt.title("Deep Tree Structure (Potential Overfitting)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_complex, y_complex, test_size=0.2, random_state=42)

for depth in [1, 3, 5, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Max depth = {depth}, Test Accuracy = {acc:.2f}")




