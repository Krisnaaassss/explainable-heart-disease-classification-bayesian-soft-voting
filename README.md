# Explainable Heart Disease Classification Using a Bayesian-Optimized Weighted Soft Voting Ensemble

This repository contains the implementation and experimental results of a machine learning framework for heart disease classification using Bayesian-optimized individual classifiers and a weighted Soft Voting Ensemble.

The proposed ensemble combines Support Vector Machine (SVM), Random Forest, and LightGBM. Model hyperparameters and ensemble voting weights are optimized using Bayesian Optimization within a stratified nested 5-fold cross-validation framework.

The framework is evaluated across five publicly available heart disease datasets. Additional resampling experiments are conducted on the two datasets with substantial class imbalance.

## Overview

This study evaluates the performance and consistency of machine learning models for heart disease classification across multiple datasets.

The experimental framework includes:

- Eight individual machine learning classifiers
- Bayesian hyperparameter optimization
- Weighted Soft Voting Ensemble
- Bayesian optimization of ensemble voting weights
- Stratified nested 5-fold cross-validation
- Leakage-aware preprocessing
- Multiple resampling strategies for imbalanced datasets
- SHAP-based feature interpretation for the main classification experiments

All preprocessing, model optimization, and resampling operations are performed within the training folds to prevent data leakage.

## Datasets

Five publicly available heart disease datasets are used:

- Hungarian
- Long Beach VA
- Statlog Heart
- Heart Failure Clinical Records
- Cleveland

The datasets differ in sample size, feature characteristics, missing values, and class distribution.

Resampling experiments are specifically performed on:

- Long Beach VA
- Heart Failure Clinical Records

These datasets have the highest class imbalance ratios among the five evaluated datasets.

## Machine Learning Models

The following classifiers are evaluated:

- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Random Forest
- Decision Tree (J48)
- Logistic Regression
- XGBoost
- LightGBM
- K-Nearest Neighbors (KNN)
- Weighted Soft Voting Ensemble

## Bayesian Hyperparameter Optimization

Bayesian Optimization is used to tune the hyperparameters of the individual classifiers.

The optimization process is performed within the inner folds of the nested cross-validation framework using F1-score as the optimization objective.

The experimental setup uses:

- 5 outer folds for model evaluation
- 5 inner folds for hyperparameter optimization
- Bayesian Optimization using `BayesSearchCV`

This separation ensures that the outer test folds are not involved in model selection or hyperparameter tuning.

## Weighted Soft Voting Ensemble

The proposed ensemble combines three heterogeneous base classifiers:

- Support Vector Machine
- Random Forest
- LightGBM

Each classifier is independently optimized using Bayesian Optimization.

The ensemble combines the predicted class probabilities of the three base learners using optimized voting weights.

The voting weights are determined using Bayesian Optimization based on out-of-fold probability predictions from the training data.

This design allows each classifier to contribute differently to the final prediction according to its predictive performance.

## Resampling Experiments

Four resampling strategies are evaluated for Long Beach VA and Heart Failure Clinical Records:

- No Resampling
- SMOTE
- Borderline-SMOTE
- SMOTE-Tomek

Resampling is performed exclusively within the training partitions of the cross-validation process.

The test folds retain their original class distributions to prevent information leakage and provide an unbiased evaluation.

### Long Beach VA

The best Soft Voting performance is obtained without resampling:

- Accuracy: 75.50%
- F1-Score: 85.80%

### Heart Failure Clinical Records

No Resampling provides the highest Accuracy:

- Accuracy: 84.60%
- F1-Score: 74.07%

SMOTE-Tomek provides the highest F1-Score:

- Accuracy: 83.60%
- F1-Score: 75.40%

The resampling results demonstrate that the effectiveness of class balancing techniques depends on the characteristics of each dataset.

## Explainability

SHAP is used in the main multi-dataset experiment to analyze feature contributions to model predictions.

The SHAP analysis is based on the LightGBM component of the proposed ensemble.

Important predictors identified across the datasets include:

- Chest pain type (`cp`)
- Number of major vessels (`ca`)
- Thalassemia status (`thal`)
- Oldpeak
- Ejection fraction
- Serum creatinine
- Serum sodium

The dedicated resampling notebook does not perform SHAP analysis because its purpose is limited to comparing resampling strategies.

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Mean performance and standard deviation are calculated across the five outer folds.

## Repository Structure

```text
.
├── Heart_Disease_Classification_All_Datasets.ipynb
├── Resampling_Dataset_Long_Beach_VA_and_Heart_Failure.ipynb
└── README.md
