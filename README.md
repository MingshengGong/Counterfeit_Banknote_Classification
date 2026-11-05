# Counterfeit Banknote Classification

**Machine Learning | Feature Selection | Classification Risk Analysis**

## Project Overview

This project develops and evaluates machine learning models to distinguish genuine banknotes from counterfeit banknotes using six numerical measurements.

The analysis focuses on predictive performance, feature efficiency, and the operational trade-off between accepting counterfeit banknotes and incorrectly rejecting genuine banknotes.

## Dataset

- **Source:** Modified Fake Bills dataset provided through University of Toronto MIE1626 teaching materials
- **Original records:** 1,500
- **Records after cleaning:** 1,463
- **Features:** Six numerical banknote measurements
- **Target:** Genuine (1) or Counterfeit (0)
- **Train/Test Split:** Stratified 70/30 split

## Methodology

1. **Data Cleaning & EDA:** Examined missing values, class distributions, feature relationships, and correlations.
2. **Model Comparison:** Evaluated KNN, Decision Tree, Random Forest, and Bagging classifiers using stratified five-fold cross-validation.
3. **Hyperparameter Optimization:** Applied GridSearchCV and RandomizedSearchCV to KNN and Random Forest.
4. **Feature Analysis:** Compared five-feature ablation and PCA to investigate dimensionality reduction.
5. **Final Evaluation:** Evaluated predefined models on an independent test set.
6. **Risk Analysis:** Examined counterfeit false acceptance and genuine false rejection rates across classification thresholds.

## Key Results

### Held-Out Test Performance

| Model | Accuracy | Macro F1 | ROC AUC |
|---|---:|---:|---:|
| Tuned Random Forest | 98.86% | 0.987280 | 0.999350 |
| Tuned KNN (6 Features) | **99.32%** | **0.992316** | 0.995971 |
| Tuned KNN (5 Features) | 99.09% | 0.989738 | **0.999559** |

The six-feature KNN achieved the highest overall test accuracy, while Random Forest achieved the highest counterfeit recall at the default classification threshold.

### Feature Analysis

- `length` and `margin_low` had the largest impact on KNN performance when individually removed.
- Removing `diagonal` maintained the original six-feature KNN cross-validation Macro F1-score of **0.986801948**.
- PCA did not demonstrate a clear performance advantage over the original feature representation.

### Classification Risk Analysis

The Random Forest threshold sensitivity analysis demonstrated the trade-off between counterfeit detection and genuine banknote acceptance.

Using training-set out-of-fold predictions:

| Counterfeit Probability Threshold | False Acceptances | False Rejections |
|---|---:|---:|
| 0.2 | 1 | 19 |
| 0.3 | 4 | 11 |
| 0.5 | 7 | 1 |

Lowering the counterfeit detection threshold reduced the number of counterfeit banknotes incorrectly accepted but increased the number of genuine banknotes incorrectly rejected.

These findings illustrate why classification accuracy alone may be insufficient for risk-sensitive decisions.

## Tools & Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Project Notebook

[View the Complete Analysis](Counterfeit_Banknote_Classification.ipynb)

## Limitations

- The dataset is relatively small and may not represent real-world banking conditions.
- Missing observations were removed using complete-case analysis.
- Final performance was evaluated on a single held-out test split.
- Probability estimates were not formally calibrated.
- Threshold analysis was exploratory; operational thresholds require independent validation and reliable error-cost estimates.

## Conclusion

The project demonstrates an end-to-end classification workflow that combines exploratory analysis, model development, feature reduction, and risk-sensitive evaluation.

The results highlight the importance of considering both predictive accuracy and the operational consequences of classification errors.
