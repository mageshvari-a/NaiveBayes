"""
Business_problem
Business Objective:
Build a model to predict if a person's salary is >50K or <=50K using demographic and work-related data.

Constraints:
- Categorical features need encoding
- Class imbalance present
- Model must be deployable via Streamlit


Data_dictionary

Feature           | Type       | Description                            | Relevant
------------------|------------|----------------------------------------|---------
age               | Numeric    | Age of the person                      | Yes
workclass         | Categorical| Type of employment                     | Yes
education         | Categorical| Education level                        | Yes
educationno       | Numeric    | Encoded education level                | Yes
maritalstatus     | Categorical| Marital status                         | Yes
occupation        | Categorical| Job role                               | Yes
relationship      | Categorical| Family relationship                    | Yes
race              | Categorical| Race                                   | Yes
sex               | Categorical| Gender                                 | Yes
capitalgain       | Numeric    | Capital income                         | Yes
capitalloss       | Numeric    | Capital loss                           | Yes
hoursperweek      | Numeric    | Weekly working hours                   | Yes
native            | Categorical| Country of origin                      | Yes
Salary            | Categorical| Target: >50K or <=50K                  | Yes

"""


import joblib  # For saving and loading trained models and encoders
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.naive_bayes import CategoricalNB, MultinomialNB, GaussianNB  # Different Naive Bayes models
import pandas as pd  # For data manipulation and analysis
from sklearn.metrics import accuracy_score, classification_report  # For evaluating model performance
from imblearn.over_sampling import SMOTE  # For handling class imbalance in training data
from collections import Counter  # For counting class occurrences
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_data = pd.read_csv("C:/CDS/Data Science/Assignment DS/10 Naive Bayes/Naive bayes Data Set/SalaryData_Train.csv")
test_data = pd.read_csv("C:/CDS/Data Science/Assignment DS/10 Naive Bayes/Naive bayes Data Set/SalaryData_Test.csv")

# Display basic info about the datasets
print("Train Data Shape:", train_data.shape)  # Prints the shape (rows, columns) of the training dataset
print("Test Data Shape:", test_data.shape)  # Prints the shape (rows, columns) of the testing dataset

# Encode categorical variables
label_encoders = {}  # Dictionary to store label encoders for consistency

for column in train_data.columns:
    if train_data[column].dtype == 'object':  # Check if the column is categorical
        le = LabelEncoder()  # Create a label encoder
        train_data[column] = le.fit_transform(train_data[column])  # Fit and transform training data
        test_data[column] = le.transform(test_data[column])  # Transform test data using the same encoder
        label_encoders[column] = le  # Store encoder for future use

# Save the label encoders for future use (e.g., during model deployment)
joblib.dump(label_encoders, "label_encoders.pkl")

# Split features and target
X_train = train_data.drop(columns=["Salary"])  # Independent variables for training
y_train = train_data["Salary"]  # Target variable for training
X_test = test_data.drop(columns=["Salary"])  # Independent variables for testing
y_test = test_data["Salary"]  # Target variable for testing

# Check class distribution before balancing
print("Before Balancing:", Counter(y_train))  # Prints class distribution before applying SMOTE

### Apply SMOTE to balance classes ###
smote = SMOTE(sampling_strategy=0.75, random_state=42)  # Adjusts class balance by oversampling the minority class
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)  # Apply SMOTE to training data

# Check class distribution after SMOTE
print("After SMOTE Balancing:", Counter(y_train_bal))  # Prints new class distribution


# Tune MultinomialNB
print("\nTuning MultinomialNB:")
alphas = [0.5, 1.0, 1.5]
for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    print(f"Alpha: {alpha}, Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Tune GaussianNB
print("\nTuning GaussianNB:")
smoothing_values = [1e-9, 1e-8, 1e-7]
for smoothing in smoothing_values:
    model = GaussianNB(var_smoothing=smoothing)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    print(f"Smoothing: {smoothing}, Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# Define different Naive Bayes models to try
models = {
    "CategoricalNB": CategoricalNB(),  # For categorical data
    "MultinomialNB": MultinomialNB(),  # For discrete count data (e.g., text classification)
    "GaussianNB": GaussianNB()  # For continuous data following a Gaussian distribution
}

best_model = None  # Variable to store the best model
best_accuracy = 0  # Track the highest accuracy achieved

for name, model in models.items():
    # Train the model
    model.fit(X_train_bal, y_train_bal)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Model: {name}, Accuracy: {accuracy:.4f}")  # Print model name and accuracy
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))  # Detailed evaluation report
    
    # Select the best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model for deployment in a Streamlit application
joblib.dump(best_model, "naive_bayes_model.pkl")

# Make final predictions using the best model
y_pred_final = best_model.predict(X_test)

# Save the test predictions to a CSV file
test_data["Predicted_Salary"] = y_pred_final  # Add predicted salary column
test_data.to_csv("test_predictions_improved_NB.csv", index=False)  # Save without index column

print("Predictions saved to test_predictions_improved_NB.csv")  # Confirm prediction file is saved
print("Model and Encoders saved successfully for deployment.")  # Confirm model saving

# Visualizations
# Univariate Plots
plt.figure(figsize=(8, 5))
sns.histplot(train_data["age"], bins=20, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=train_data, x="education", order=train_data['education'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Education Level Counts")
plt.ylabel("Count")
plt.show()

# Bivariate Plots
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_data, x="Salary", y="age")
plt.title("Age vs Salary")
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(data=train_data, x="occupation", hue="Salary", order=train_data['occupation'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Occupation vs Salary")
plt.ylabel("Count")
plt.show()

# Business Impact Summary
"""
The Business Impact:
Better Classification: By using the Naive Bayes classifiers and balancing the dataset with SMOTE, the model 
can more effectively predict salary levels, ensuring accuracy across different demographic groups.

Scalability: The model can be deployed for real-time predictions in a job application system, improving hiring 
decisions based on salary predictions.

Cost Efficiency: With a lightweight and efficient algorithm, Naive Bayes models can be quickly retrained or 
deployed, making them suitable for production environments with limited resources.

"""

