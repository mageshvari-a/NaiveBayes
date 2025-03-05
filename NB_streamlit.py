import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load the trained model and label encoders using relative paths
model_path = os.path.join(os.path.dirname(__file__), "naive_bayes_model.pkl")
encoders_path = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")

model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# Define the input fields
st.title("Salary Prediction App")
st.markdown("### Enter details to predict if salary is >50K or <=50K")

# List of features based on the dataset (excluding 'Salary' as it's the target variable)
features = [
    "age", "workclass", "education", "educationno", "maritalstatus",
    "occupation", "relationship", "race", "sex", "capitalgain",
    "capitalloss", "hoursperweek", "native"
]

# Create input fields dynamically
user_input = {}
for feature in features:
    if feature in label_encoders:  # Categorical input
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(f"Select {feature}", options)
    else:  # Numerical input
        user_input[feature] = st.number_input(f"Enter {feature}", step=1)

# Convert user input into a dataframe
user_df = pd.DataFrame([user_input])

# Encode categorical values (excluding 'Salary' since it's the target variable)
for feature in label_encoders:
    if feature in user_df.columns:
        try:
            user_df[feature] = label_encoders[feature].transform(user_df[feature])
        except ValueError as e:
            print(f"Encoding error for {feature}: {e}")
    elif feature != "Salary":  # Ignore warning for 'Salary'
        print(f"Warning: {feature} not found in user_df.")

# Debugging: Print DataFrame columns and Label Encoders used
print("DataFrame columns:", user_df.columns)
print("Label Encoders:", label_encoders.keys())

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(user_df)
    salary_label = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"Predicted Salary: {salary_label}")
