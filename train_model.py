import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ---------------------------------------------------------
# 1. Load the Data
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'phishing_features.csv')
model_dir = os.path.join(script_dir, 'models')

# Create a folder to store our trained model if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# Separate our features (X) from our target label (y)
# Drop the 'label' column to get X, and keep only the 'label' column for y
X = df.drop('label', axis=1)
y = df['label']

# ---------------------------------------------------------
# 2. Train-Test Split
# ---------------------------------------------------------
# We split the data: 80% to train the model, 20% to test it on data it has NEVER seen before.
# This proves the model is actually learning, not just memorizing.
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. Train the Model
# ---------------------------------------------------------
print("Training the Random Forest Classifier (this may take a few seconds)...")
# Initialize the model with 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Fit (Train) the model on the training data
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Evaluate the Model
# ---------------------------------------------------------
print("\nEvaluating model performance on test data...")
# Ask the model to predict labels for the 20% of data it was not trained on
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- MODEL ACCURACY: {accuracy * 100:.2f}% ---\n")

# Print detailed metrics (Precision, Recall, F1-Score)
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Phishing (1)"]))

# ---------------------------------------------------------
# 5. Analyze Feature Importance
# ---------------------------------------------------------
# Which feature was the biggest giveaway that a domain was phishing?
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop Predictors of a Phishing Domain:")
print(feature_importance_df.to_string(index=False))

# ---------------------------------------------------------
# 6. Save the Model
# ---------------------------------------------------------
model_path = os.path.join(model_dir, 'phishing_rf_model.pkl')
print(f"\nSaving trained model to: {model_path}")
joblib.dump(rf_model, model_path)
print("Success! Model is saved and ready for the API.")