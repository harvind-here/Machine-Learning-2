# Machine Learning Predictive Models

This repository contains implementations of multiple machine learning models applied to various datasets. Each model has been evaluated using performance metrics like accuracy and confusion matrices, and the results are stored in the `results` folder.

## Datasets

The following datasets have been used in this project:
- **Iris Dataset**: A classification dataset for predicting the species of Iris flowers.
- **Breast Cancer Dataset**: A classification dataset for identifying malignant vs. benign tumors.
- **Wine Dataset**: A classification dataset for predicting wine quality based on chemical composition.
- **Digits Dataset**: A multi-class classification dataset for handwritten digit recognition (0-9).

## Models

The following machine learning models were applied to each dataset:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

For each model, we trained, evaluated, and plotted confusion matrices to visualize performance. You can find the output plots in the [`results`](./results) folder.

## Project Structure
. 
├── data/ # Dataset loading and preprocessing 
├── results/ # Plots of confusion matrices for each model 
├── model_training.py # Main script to train and evaluate models 
├── requirements.txt # Dependencies and libraries 
└── README.md # Project overview and instructions

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ml-predictive-models.git
cd machine-learning-2
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Run the project to train models and generate results:
```bash
python model_training.py
```

