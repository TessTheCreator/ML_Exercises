# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Set a random seed
import random
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(features_raw.head())

# Removing the names
features_no_names = features_raw.drop(['Name'], axis=1)

# One-hot encoding
features = pd.get_dummies(features_no_names)

features = features.fillna(0.0)
display(features.head())

X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)

def original_model() :
    # Define the classifier, and fit it to the data
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Making predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the accuracy
    from sklearn.metrics import accuracy_score
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('The training accuracy is', train_accuracy)
    print('The test accuracy is', test_accuracy)

def optimized_model() :

    # Train the model
    param_grid = {
        'max_depth': [5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 4, 7, 6, 8, 10, 12, 14, 16, 18, 20],
        'min_samples_leaf': [6, 7, 8, 9, 10]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    new_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = new_model.predict(X_train)
    y_test_pred = new_model.predict(X_test)

    # Calculate the accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Best parameters:", grid_search.best_params_)
    print('The training accuracy of the new model is', train_accuracy)
    print('The test accuracy of the new model is', test_accuracy)


original_model()
optimized_model()

