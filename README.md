# Exoplanet-Detection
This machine learning project aims to detect exoplanets in stellar light curves using the K-Nearest Neighbors (KNN) algorithm. The dataset consists of flux measurements at different time points for various stars, with labels indicating the presence (1) or absence (0) of exoplanets.

Key Steps:

Data Exploration: Analyzed dataset shape, checked for null values, and visualized label distribution.
Data Preprocessing: Replaced labels (2 with 1, 1 with 0) for binary classification. Handled outliers using KNN sensitivity.
Feature Scaling: Applied standardization to independent features for model compatibility.
Model Training: Utilized KNN classifier with 5 neighbors and Euclidean distance metric.
Model Evaluation: Assessed the model's performance using accuracy, classification report, confusion matrix, and ROC-AUC.
This project demonstrates the application of machine learning to identify exoplanets in stellar observations, emphasizing data preprocessing and model evaluation for robust results.

Libraries Used: pandas, seaborn, numpy, matplotlib, scikit-learn
