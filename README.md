# Counterfeit Banknote Classification Using Machine Learning

## Project Overview

This project develops and evaluates supervised machine learning models to distinguish genuine banknotes from counterfeit banknotes using physical measurement features.

The objective is to investigate different classification algorithms, evaluate the impact of feature selection and dimensionality reduction, and identify an effective model through cross-validation and hyperparameter optimization.

## Key Results

| Metric | Result |
|---|---|
| Best Model | K-Nearest Neighbors (KNN) |
| Test Accuracy | 98.86% |
| ROC AUC | 0.9908 |
| Selected Features | 5 |
| Optimal Number of Neighbors | 6 |

The optimized KNN classifier demonstrates strong classification performance on the held-out test dataset.

## Methods and Techniques

- **Exploratory Data Analysis (EDA):** Data visualization, descriptive statistics, and feature relationships.
- **Data Preprocessing:** Feature standardization using StandardScaler.
- **Feature Selection:** Forward selection and decision tree-based feature importance.
- **Dimensionality Reduction:** Principal Component Analysis (PCA).
- **Classification Models:** KNN, Decision Tree, Random Forest, and Bagging.
- **Hyperparameter Optimization:** GridSearchCV, RandomizedSearchCV, and 5-fold cross-validation.
- **Model Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC.

## Tools and Technologies

Python · Pandas · NumPy · Matplotlib · Scikit-learn · Jupyter Notebook

## Project Notebook

The complete project, including Python code, visualizations, model comparisons, and evaluation results, is available here:

[View Complete Jupyter Notebook](Counterfeit_Banknote_Classification.ipynb)

## Key Takeaways

This project demonstrates an end-to-end supervised machine learning workflow, from data exploration and preprocessing to feature selection, model optimization, and performance evaluation.

It highlights the importance of systematically comparing classification algorithms and assessing model performance using multiple evaluation metrics.
