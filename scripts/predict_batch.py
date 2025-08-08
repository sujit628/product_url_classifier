import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from utils.preprocessing import prepare_text_features

# Load data
df = pd.read_csv("data/urls_dataset.csv")

# Preprocess
X_text = prepare_text_features(df)

# Load model and vectorizer
model = joblib.load("models/url_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Transform and predict
X_vec = vectorizer.transform(X_text)
df['predicted_category'] = model.predict(X_vec)

# Save predictions
os.makedirs("output", exist_ok=True)
df.to_csv("output/predictions.csv", index=False)

print("✅ Predictions saved to output/predictions.csv")
