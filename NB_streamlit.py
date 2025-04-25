import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and label encoders
model = joblib.load("naive_bayes_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

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

# Display the age distribution
st.subheader("Distribution of Age")
plt.figure(figsize=(8, 5))
sns.histplot(train_data["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
st.pyplot(plt)

# Display education level counts
st.subheader("Education Level Counts")
plt.figure(figsize=(10, 5))
sns.countplot(data=train_data, x="education", order=train_data['education'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Education Level Counts")
plt.ylabel("Count")
st.pyplot(plt)

# Display Age vs Salary boxplot
st.subheader("Age vs Salary")
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_data, x="Salary", y="age")
plt.title("Age vs Salary")
st.pyplot(plt)

# Display Occupation vs Salary countplot
st.subheader("Occupation vs Salary")
plt.figure(figsize=(12, 5))
sns.countplot(data=train_data, x="occupation", hue="Salary", order=train_data['occupation'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Occupation vs Salary")
plt.ylabel("Count")
st.pyplot(plt)

# Debugging: Print DataFrame columns and Label Encoders used
print("DataFrame columns:", user_df.columns)
print("Label Encoders:", label_encoders.keys())

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(user_df)
    salary_label = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"Predicted Salary: {salary_label}")
