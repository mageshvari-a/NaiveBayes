import joblib  # For saving model and encoders
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB, MultinomialNB, GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # For balancing dataset
from collections import Counter
from sklearn.model_selection import train_test_split

# Load the datasets
train_data = pd.read_csv("C:/CDS/Data Science/Assignment DS/10 Naive Bayes/Naive bayes Data Set/SalaryData_Train.csv")
test_data = pd.read_csv("C:/CDS/Data Science/Assignment DS/10 Naive Bayes/Naive bayes Data Set/SalaryData_Test.csv")

# Display basic info
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# Encode categorical variables
label_encoders = {}  # Store encoders for consistency between train and test sets

for column in train_data.columns:
    if train_data[column].dtype == 'object':  # If categorical
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])  # Apply same encoding
        label_encoders[column] = le  # Store encoder for later use

# Save the label encoders for future use in deployment
joblib.dump(label_encoders, "label_encoders.pkl")

# Split features and target
X_train = train_data.drop(columns=["Salary"])  # Independent variables
y_train = train_data["Salary"]  # Target variable
X_test = test_data.drop(columns=["Salary"])
y_test = test_data["Salary"]

# Check class distribution before balancing
print("Before Balancing:", Counter(y_train))

### Apply SMOTE to balance classes ###
smote = SMOTE(sampling_strategy=0.75, random_state=42)  # Adjust ratio as needed
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("After SMOTE Balancing:", Counter(y_train_bal))

# Try different Naive Bayes variants
models = {
    "CategoricalNB": CategoricalNB(),
    "MultinomialNB": MultinomialNB(),
    "GaussianNB": GaussianNB()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    # Train the model
    model.fit(X_train_bal, y_train_bal)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {name}, Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
    
    # Select the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model for Streamlit deployment
joblib.dump(best_model, "naive_bayes_model.pkl")

# Make final predictions
y_pred_final = best_model.predict(X_test)

# Save the predictions
test_data["Predicted_Salary"] = y_pred_final
test_data.to_csv("test_predictions_improved.csv", index=False)
print("Predictions saved to test_predictions_improved.csv")
print("Model and Encoders saved successfully for deployment.")
