# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess dataset
def load_and_preprocess_dataset(dataset_func):
    dataset = dataset_func()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target

    # Handle missing values (if applicable)
    X.fillna(X.mean(), inplace=True)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function for training and evaluating models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nModel: {name} on {dataset_name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        # Visualization of Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} on {dataset_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

# Datasets to work with
datasets = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
    "Digits": load_digits
}

# Iterate through datasets
for dataset_name, dataset_func in datasets.items():
    print(f"\n--- Processing {dataset_name} Dataset ---\n")
    
    # Load and preprocess the dataset
    X_scaled, y = load_and_preprocess_dataset(dataset_func)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name)

# Optional: Hyperparameter tuning for K-Nearest Neighbors
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

# Use the Breast Cancer dataset for example
X_bc, y_bc = load_and_preprocess_dataset(load_breast_cancer)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.3, random_state=42)
grid.fit(X_train_bc, y_train_bc)
print("Best Params for KNN (Breast Cancer Dataset):", grid.best_params_)

# Visualization for Wine Dataset
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_bc, grid.predict(X_test_bc)), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - KNN with Hyperparameter Tuning on Breast Cancer Dataset')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
