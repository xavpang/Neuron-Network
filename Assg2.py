# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:01:32 2024

@author: xavierpang
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'Depression.csv'
data = pd.read_csv(file_path)

# Define categorical and numerical columns
categorical_columns = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                       'Have you ever had suicidal thoughts ?', 
                       'Family History of Mental Illness']
numerical_columns = ['Age', 'Academic Pressure', 
                     'CGPA', 'Study Satisfaction', 
                     'Work/Study Hours', 
                     'Financial Stress']

# Numerical columns: Fill missing values with the median
num_imputer = SimpleImputer(strategy='median')
data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])

# Encode categorical data
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for future use

# Define features (X) and target (y)
X = data[categorical_columns + numerical_columns]
y = data['Depression']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the ANN model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=42, learning_rate_init=0.05)
mlp.fit(X_train, y_train)

# Predictions and evaluation
predictions = mlp.predict(X_test)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Display results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Neural network visualization function
def visualise(mlp):
    # Each layer's neuron count
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    # Neuron coordinates
    loc_neurons = [[[l, (n + 1) * (1 / (layer + 1))] for n in range(layer)] for l, layer in enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x, y in layer]
    y_neurons = [y for layer in loc_neurons for x, y in layer]

    # Weight range
    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
    for l, layer in enumerate(mlp.coefs_):
        for i, neuron in enumerate(layer):
            for j, w in enumerate(neuron):
                ax.plot(
                    [loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],
                    [loc_neurons[l][i][1], loc_neurons[l + 1][j][1]],
                    color='grey',
                    linewidth=(w - weight_range[0]) / (weight_range[1] - weight_range[0]) * 5 + 0.2,
                )
    plt.show()

# Visualize the ANN
visualise(mlp)