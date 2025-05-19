# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas_profiling

# Load the dataset
data = pd.read_csv('customer_churn_data.csv')  # Replace with your dataset path
print(data.head())

# Exploratory Data Analysis (EDA)
# Generate a profile report
profile = pandas_profiling.ProfileReport(data)
profile.to_file("customer_churn_report.html")  # Save the report

# Visualizing the distribution of the target variable
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Data Preprocessing
# Handling missing values
data.fillna(data.mean(), inplace=True)  # Impute missing values with mean for numerical columns

# Encoding categorical variables
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numerical_features = data.select_dtypes(exclude=['object']).columns.tolist()

# Splitting the dataset into features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Model Building
# Logistic Regression Model
logistic_model = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression())])

logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Random Forest Model
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
# Logistic Regression Evaluation
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))
print("Logistic Regression ROC AUC Score:", roc_auc_score(y_test, y_pred_logistic))

# Random Forest Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(y_test, y_pred_logistic, 'Logistic Regression')
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest')

# Deployment with Streamlit
# Uncomment the following lines to run a Streamlit app
# import streamlit as st
# st.title('Customer Churn Prediction')
# user_input = st.text_input("Enter customer details here...")
# # Add code to process user input and make predictions
# st.write("Prediction: ", prediction)

# Deployment with Gradio
# Uncomment the following lines to run a Gradio interface
# import gradio as gr
# def predict_churn(input_data):
#     # Process input_data and make predictions
#     return prediction
# gr.Interface(fn=predict_churn, inputs="text", outputs="text").launch()
