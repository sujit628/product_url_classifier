import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
import joblib

# Setup path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import prepare_text_features

# --- Load and check dataset ---
DATA_PATH = "data/urls_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
required_cols = ['url', 'title', 'category']

if not all(col in df.columns for col in required_cols):
    raise ValueError(f"❌ Dataset must contain columns: {required_cols}")

# --- Prepare features and labels ---
X_text = prepare_text_features(df)
y = df['category']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(X_text)

# Check label balance
label_counts = y.value_counts()
if any(label_counts < 2):
    print(f"⚠️ Some classes have fewer than 2 samples. Skipping stratify.")
    stratify_param = None
else:
    stratify_param = y

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=stratify_param, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/url_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model training complete.")
