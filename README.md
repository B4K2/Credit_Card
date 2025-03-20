# Credit Card Fraud Detection System

## Overview

This project is a **Credit Card Fraud Detection System** that utilizes machine learning techniques to identify fraudulent transactions. The model is trained on a dataset containing transaction details and classifies each transaction as either legitimate or fraudulent.

## Features

- **Preprocessing**: Handles missing values, scales numerical data, and encodes categorical variables.
- **Feature Selection**: Uses Mutual Information Score to select relevant features and drops non-recommended fields.
- **Model Training**: Implements multiple machine learning models:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - XGBoost
  - Each model is trained in two variations: without hyperparameter tuning and with hyperparameter tuning.
- **Evaluation Metrics**: Uses accuracy, precision, recall, and F1-score to assess model performance.
- **Visualization**: Includes graphs and plots to analyze data distribution and model results.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- XGBoost

## Installation

1. Clone the repository:
2. Navigate to the project directory:
3. Install dependencies:

## Usage

1. Open the Jupyter Notebook:
2. Run the `Untitled.ipynb` file step by step to preprocess data, train models, and evaluate results.

## Dataset

- The dataset contains transaction details with features such as transaction amount, time, and anonymized variables.
- Labels: **0 (Legitimate)**, **1 (Fraudulent)**
- [Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- The dataset contains transaction details with features such as transaction amount, time, and anonymized variables.
- Labels: **0 (Legitimate)**, **1 (Fraudulent)**

## Results

### Random Forest Classifier (With Tuning)

- **Accuracy**: 0.9995318501494888

- **F1-score**: 0.8543046357615894

- **Recall**: 0.8657718120805369

- **Precision**: 0.8431372549019608

- The model achieves high accuracy in detecting fraudulent transactions while minimizing false positives.

- Feature importance analysis helps understand the most influential factors in fraud detection.

## Future Improvements

- Implement deep learning models for better fraud detection.
- Deploy the model using Flask or FastAPI.
- Integrate with a real-time transaction monitoring system.

## Contributing

Feel free to fork the repository and contribute improvements via pull requests

