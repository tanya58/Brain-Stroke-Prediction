# Brain-Stroke-Prediction
# Dataset link - https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
# Python Libraries Used:
pandas – For data loading, preprocessing, and manipulation
import pandas as pd

numpy – For numerical operations (if used)
import numpy as np

matplotlib – For data visualization (e.g., plt.figure, plt.show)
import matplotlib.pyplot as plt

seaborn – For advanced statistical visualizations (e.g., sns.countplot)
import seaborn as sns

scikit-learn – For machine learning models and evaluation

# Preprocessing
• Conducted comprehensive data preprocessing:
– Checked for missing values and duplicates using df.isnull().sum() and df.duplicated().sum()
– Identified and explored categorical features (dtype == 'O') for further processing
– Assessed null entries in categorical columns and handled them for clean model input

# EDA - Exploratory Data Analysis
Visualized distributions of categorical and numerical features using pie charts and histograms to identify patterns and imbalances.

Analyzed relationships between input features and the target variable using count plots and bar charts to explore potential correlations.

Created a heatmap to examine feature-to-feature correlation. The correlation matrix revealed minimal linear relationships among attributes, suggesting the need for non-linear models for accurate prediction.

# Model Building
Creating a train and test split of the oversampled dataset. (80-20)
Applied various Machine learning models for predictive analysis

Decision tree
KNN
SVM
Random forest
Logistic regression
Analysed the results generated using confusion matrix - accuracy, precision, recall and plotting the ROC plot and generating the AUC scores.

Accuracies calculated:

Decision tree : 91.096 %
KNN : 93.73%
SVM : 93.93%
Random forest : 93.836 %
Logistic regression : 93.933 %

Highest accuracy (93.933%)

Tied very closely with SVM (93.93%), but still slightly ahead
