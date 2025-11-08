# Heart Disease Prediction App

A web application built with **Streamlit** for predicting heart disease based on patient data using a **Logistic Regression** model. The app provides interactive visualizations to explore the dataset, model predictions, and evaluation metrics.

---

## Features

- **Prediction**  
  - Enter patient data and get the probability of heart disease.
  - Adjust prediction thresholds and visualize the impact.

- **Model Evaluation**  
  - Display key metrics: F1 Score, Precision, Recall, Accuracy, Log Loss, and ROC AUC.
  - Confusion matrix with heatmap and annotations for True/False Positives/Negatives.
  - ROC Curve visualization.
  
- **Data Exploration**  
  - Histograms showing the distribution of predicted probabilities.
  - Scatter plots comparing predictions with actual outcomes.
  - Feature-based analysis: visualize prediction probabilities and correctness by feature (e.g., age, cholesterol).

- **Interactive Visualizations**  
  - Probability distribution per class.
  - Correct vs incorrect predictions.
  - Feature importance exploration with scatter plots.

- **Preprocessing**  
  - **OneHotEncoder** for categorical variables (transforms categories into binary columns).  
  - **StandardScaler** for numerical features (standardizes features to mean=0 and std=1).

- **Hyperparameter Tuning**  
  - Model was trained using **GridSearchCV** to select the best parameters for Logistic Regression, improving performance and avoiding overfitting.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd heart_disease_app
