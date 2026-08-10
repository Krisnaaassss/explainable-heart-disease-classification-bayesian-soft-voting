# Heart Disease Classification with Resampling and Ensemble

This repository contains machine learning experiments for heart disease classification using multiple datasets, Bayesian hyperparameter optimization, ensemble learning, and resampling techniques.

## Overview

The experiments evaluate several machine learning models across multiple heart disease datasets. Model performance is assessed using nested cross-validation, while Bayesian optimization is used for hyperparameter tuning.

A Soft Voting Ensemble is also evaluated by combining selected base learners.

For imbalanced datasets, additional resampling experiments are conducted to compare different class balancing techniques.

## Datasets

The experiments use the following datasets:

- Hungarian
- Long Beach VA
- Statlog Heart
- Heart Failure Clinical Records
- Cleveland

Resampling experiments are specifically conducted on:

- Long Beach VA
- Heart Failure Clinical Records

## Machine Learning Models

The evaluated models include:

- Support Vector Machine
- Naive Bayes
- Random Forest
- Decision Tree
- Logistic Regression
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Soft Voting Ensemble

## Hyperparameter Optimization

Bayesian Optimization is used to determine suitable hyperparameters for the machine learning models.

The experiments use nested 5-fold cross-validation to separate hyperparameter optimization from model evaluation.

## Ensemble Method

The Soft Voting Ensemble combines:

- Support Vector Machine
- Random Forest
- LightGBM

The ensemble combines predicted probabilities from the base learners using optimized voting weights.

## Resampling Methods

The following resampling strategies are evaluated on the imbalanced datasets:

- No Resampling
- SMOTE
- Borderline-SMOTE
- SMOTE-Tomek

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## Repository Structure

```text
.
├── Heart_Disease_Classification_All_Datasets.ipynb
├── Resampling_Dataset_Long_Beach_VA_and_Heart_Failure.ipynb
└── README.md
