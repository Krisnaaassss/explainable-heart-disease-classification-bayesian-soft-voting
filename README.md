# Explainable Heart Disease Classification Using a Bayesian-Optimized Weighted Soft Voting Ensemble

This repository contains machine learning experiments for heart disease classification using Bayesian optimization, a weighted Soft Voting Ensemble, nested cross-validation, and resampling techniques across multiple public datasets.

## Overview

The proposed framework combines Support Vector Machine (SVM), Random Forest, and LightGBM using optimized voting weights.

Model hyperparameters and ensemble weights are optimized using Bayesian Optimization within a stratified nested 5-fold cross-validation framework.

## Datasets

Five public heart disease datasets are evaluated:

- Hungarian
- Long Beach VA
- Statlog Heart
- Heart Failure Clinical Records
- Cleveland

Resampling experiments are specifically conducted on:

- Long Beach VA
- Heart Failure Clinical Records

## Models

The evaluated models include:

- Support Vector Machine
- Gaussian Naive Bayes
- Random Forest
- Decision Tree
- Logistic Regression
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Weighted Soft Voting Ensemble

## Methodology

The experimental framework includes:

- Bayesian hyperparameter optimization
- Stratified nested 5-fold cross-validation
- Weighted Soft Voting Ensemble
- Bayesian optimization of voting weights
- Leakage-aware preprocessing
- SHAP-based feature interpretation

The Soft Voting Ensemble combines:

- SVM
- Random Forest
- LightGBM

## Resampling Methods

The following strategies are evaluated for Long Beach VA and Heart Failure Clinical Records:

- No Resampling
- SMOTE
- Borderline-SMOTE
- SMOTE-Tomek

Resampling is applied only within the training folds to prevent data leakage.

## Evaluation Metrics

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## Main Results

The Weighted Soft Voting Ensemble achieved:

- Average Accuracy: 82.46%
- Average F1-Score: 79.82%

For the resampling experiments:

- Long Beach VA achieved its best F1-Score without resampling: 85.80%
- Heart Failure Clinical Records achieved its highest Accuracy without resampling: 84.60%
- SMOTE-Tomek achieved the highest F1-Score on Heart Failure Clinical Records: 75.40%

## Repository Structure

```text
.
├── Heart_Disease_Classification_All_Datasets.ipynb
├── Resampling_Dataset_Long_Beach_VA_and_Heart_Failure.ipynb
└── README.md
