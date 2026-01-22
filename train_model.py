import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("data/breast_cancer.csv")

# Drop unnecessary column if present
if 'id' in data.columns:
    data = data.drop(['id'], axis=1)

# Encode target: M = 1, B = 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=500))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open("../model/breast_cancer_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully.")


