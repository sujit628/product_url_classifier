import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.metrics import classification_report
from utils.preprocessing import prepare_text_features

# Load dataset
df = pd.read_csv("data/urls_dataset.csv")

# Prepare features
X_text = prepare_text_features(df)
y_true = df['category']

# Load model and vectorizer
model = joblib.load("models/url_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Transform and predict
X_vec = vectorizer.transform(X_text)
y_pred = model.predict(X_vec)

# Evaluate
report = classification_report(y_true, y_pred)
print("✅ Model Evaluation Report:\n")
print(report)
