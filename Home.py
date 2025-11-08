import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score,log_loss,precision_score,recall_score,confusion_matrix,roc_auc_score,roc_curve, auc

st.set_page_config(page_title="Dataset & Model Insights",page_icon="heart.png", layout="centered")

st.title("Dataset & Model Insights")
st.write("Explore the dataset used to train the heart disease prediction model.")


df = pd.read_csv("heart_disease_uci.csv")
pre_df = pd.read_csv("preprocessed_dataset.csv")

model = joblib.load("heart_disease_model.pkl")


st.header("Before preprocessing")
st.subheader("Dataset Overview")
st.write(df.head(10))

st.subheader("Dataset Summary")
st.write(df.describe(include='all'))

st.subheader("Dataset Shape")
st.write(df.shape)

st.header("After preprocessing")
st.subheader("Dataset Overview")
st.write(pre_df.head(10))

st.subheader("Dataset Summary")
st.write(pre_df.describe(include='all'))

st.subheader("Dataset Shape")
st.write(""" \
   As you may have noticed, the number of columns has increased.
   This is because the dataset has been processed in two main steps: encoding and scaling.
   First, OneHotEncoder was applied to categorical variables,
   transforming each category into a separate binary column, which allows the model to interpret categorical data numerically.
   Then, StandardScaler was used to scale the numerical features, standardizing them to have a mean of 0 and a standard deviation of 1.
    This ensures that all features contribute equally to the model and prevents features with larger scales from dominating the learning process.
    """)
st.write(pre_df.shape)

st.subheader("Class Distribution (Disease vs No Disease)")
fig, ax = plt.subplots()
pre_df['disease'].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel("Disease (1 = Yes, 0 = No)")
ax.set_ylabel("Count")
st.pyplot(fig)


st.header("Feature Correlation")

corr = pre_df.corr()
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr, cmap='coolwarm',annot=True,ax=ax)
ax.set_title('Correlation matrix')
st.pyplot(fig)


st.header("Feature Distributions")

selected = st.selectbox(
    "Choose a feature to visualize:",
    ['age', 'chol', 'trestbps', 'thalch', 'oldpeak'],
    key="feature_select_1"
)

fig2, ax2 = plt.subplots()
ax2.hist(pre_df[selected], bins=20)
ax2.set_xlabel(selected)
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

st.header("Model Performance")

test_df = pd.read_csv('heart_test.csv')


test_x = test_df.drop('disease', axis=1)
test_y = test_df['disease']

pred = model.predict(test_x)
pred_proba = model.predict_proba(test_x)[:,1]

st.text(f"F1 Score: {f1_score(test_y, pred):.2f}")
st.text(f"Log Loss: {log_loss(test_y, pred_proba):.4f}")
st.text(f"Precision Score: {precision_score(test_y, pred):.2f}")
st.text(f"Recall Score: {recall_score(test_y, pred):.2f}")
st.text(f"Confusion Matrix:\n{confusion_matrix(test_y, pred)}")
st.text(f"Accuracy: {model.score(test_x, test_y):.2f}")
st.text(f"ROC AUC Score: {roc_auc_score(test_y, pred_proba):.2f}")


st.subheader("ROC Curve - Heart Disease Classification")
fpr, tpr, thresholds = roc_curve(test_y, pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve - Heart Disease Classification')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

st.pyplot(fig)

st.subheader("Plot distribution of probabilities disease")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(
    data=pd.DataFrame({
        'probability': pred_proba,
        'actual': test_y.map({0: 'No Disease', 1: 'Disease'})
    }),
    x='probability',
    hue='actual',
    bins=30,
    kde=True,
    ax=axes[0]
)
axes[0].axvline(x=0.30, color='red', linestyle='--', label='Threshold (0.30)')
axes[0].set_xlabel('Predicted Probability of Disease')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Predicted Probabilities')
axes[0].legend()

pred_final = (pred_proba >= 0.30).astype(int)
comparison = pd.DataFrame({
    'Actual': test_y,
    'Predicted': pred_final,
    'Probability': pred_proba
})
comparison = comparison.sort_values('Probability')
comparison['Index'] = range(len(comparison))

colors = ['green' if a == p else 'red' for a, p in zip(comparison['Actual'], comparison['Predicted'])]
axes[1].scatter(comparison['Index'], comparison['Probability'], c=colors, alpha=0.6)
axes[1].axhline(y=0.30, color='red', linestyle='--', label='Threshold')
axes[1].set_xlabel('Patient Index (sorted by probability)')
axes[1].set_ylabel('Predicted Probability')
axes[1].set_title('Predictions: Green=Correct, Red=Wrong')
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)


st.subheader("Plot confusion matrix")
cm = confusion_matrix(test_y, pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    cbar=True,
    xticklabels=['No Disease', 'Disease'],
    yticklabels=['No Disease', 'Disease'],
    annot_kws={'size': 16},
    ax=ax
)

ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.text(0.5, -0.3, f'True Negatives: {cm[0,0]}', ha='center', fontsize=10, color='green', transform=ax.transAxes)
ax.text(1.5, -0.3, f'False Positives: {cm[0,1]}', ha='center', fontsize=10, color='orange', transform=ax.transAxes)
ax.text(0.5, 1.1, f'False Negatives: {cm[1,0]}', ha='center', fontsize=10, color='orange', transform=ax.transAxes)
ax.text(1.5, 1.1, f'True Positives: {cm[1,1]}', ha='center', fontsize=10, color='green', transform=ax.transAxes)

plt.tight_layout()
st.pyplot(fig)


st.subheader("Plot 1: Probability vs Feature & Plot 2: Correct vs Incorrect predictions")
feature_name = st.selectbox(
    "Choose a feature to visualize:",
    ['age', 'chol', 'trestbps', 'thalch', 'oldpeak'],
    key="feature_select_2"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

scatter = axes[0].scatter(
    test_x[feature_name],
    pred_proba,
    c=test_y,
    cmap='RdYlGn',
    alpha=0.6,
    s=50
)
axes[0].axhline(y=0.30, color='red', linestyle='--', label='Threshold (0.30)')
axes[0].set_xlabel(f'{feature_name}', fontsize=12)
axes[0].set_ylabel('Predicted Probability', fontsize=12)
axes[0].set_title(f'Prediction Probability vs {feature_name}')
fig.colorbar(scatter, ax=axes[0], label='Actual (0=Healthy, 1=Disease)')
axes[0].legend()
axes[0].grid(alpha=0.3)

pred_final = (pred_proba >= 0.30).astype(int)
correct = (pred_final == test_y)

axes[1].scatter(
    test_x[correct][feature_name],
    pred_proba[correct],
    c='green',
    alpha=0.6,
    s=50,
    label='Correct'
)
axes[1].scatter(
    test_x[~correct][feature_name],
    pred_proba[~correct],
    c='red',
    alpha=0.6,
    s=50,
    label='Incorrect'
)
axes[1].axhline(y=0.30, color='blue', linestyle='--', label='Threshold')
axes[1].set_xlabel(f'{feature_name}', fontsize=12)
axes[1].set_ylabel('Predicted Probability', fontsize=12)
axes[1].set_title(f'Correct vs Incorrect Predictions by {feature_name}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)


st.header("What is GridSearchCV?")

st.markdown("""
**GridSearchCV** is a technique used to automatically find the best hyperparameters for a machine learning model.

Instead of manually guessing values like *regularization strength (C)* or *penalty type* in Logistic Regression, 
GridSearchCV tries all combinations for you and uses cross-validation to evaluate performance.

### Why is it useful?
Saves time  
Prevents underfitting / overfitting  
Helps the model generalize better  
Finds optimal settings automatically  

### Example parameters tuned for Logistic Regression:
- `C`: controls regularization strength  
- `penalty`: L1 or L2
- `solver`: optimization algorithm

GridSearchCV helped us select the best combination, improving overall accuracy and consistency.
""")

st.header("Why Logistic Regression Works Well Here")

st.markdown("""
Logistic Regression is ideal for medical binary outcome prediction because:

It predicts probabilities  
Easy to interpret  
Not computationally heavy  
Works well with fewer features  
Good baseline model in medical diagnosis

The output is the probability of heart disease (0 to 100%).
""")
