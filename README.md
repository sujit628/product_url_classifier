🧠 ML Model: Product URL Classification

Classify any given URL into one of the following:

    product – Direct product page

    category – Listing or category page

    other – Blog post, seller profile, etc.

📁 Project Structure

product_url_classifier/
├── data/                      # Input data (CSV)
│   └── urls_dataset.csv
├── models/                    # Saved model and vectorizer
│   ├── url_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── output/                    # Prediction results
│   └── predictions.csv
├── scripts/                   # Main scripts
│   ├── train_model.py
│   ├── predict_batch.py
│   └── evaluate_model.py
├── utils/                     # Reusable functions
│   ├── __init__.py
│   └── preprocessing.py
├── requirements.txt           # Python dependencies
└── README.md                  # This guide

✅ Prerequisites

    Python 3.10 recommended (not 3.13!)

    VS Code or any IDE/terminal

⚙️ Setup Instructions
1. Clone or Download the Project

git clone <your-repo-url>
cd product_url_classifier

2. Create & Activate Virtual Environment (Windows)

python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

    ⚠️ If scikit-learn fails, make sure you're using Python 3.10

📊 Dataset

Put your labeled URL dataset in:

data/urls_dataset.csv

Required columns:

url,title,category
https://www.amazon.in/product123,Buy Now,product
https://www.flipkart.com/mobiles,Mobile Listings,category
https://blog.example.com/tips,Top 10 Tips,other

✅ At least 2 samples per class required to train properly.

🚀 How to Run

🔹 1. Train the Model

python scripts/train_model.py

This:

    Trains the model

    Saves url_classifier.pkl and tfidf_vectorizer.pkl in models/

🔹 2. Predict on Dataset

python scripts/predict_batch.py

This:

    Adds predicted_category to each row

    Saves output to output/predictions.csv

🔹 3. Evaluate Model Performance

python scripts/evaluate_model.py

This:

    Prints accuracy, precision, recall, and F1-score

📦 Retraining

To retrain with a new dataset:

    Replace data/urls_dataset.csv

    Re-run:

python scripts/train_model.py

🧠 Technologies Used

    Python 3.10

    scikit-learn

    pandas

    TF-IDF (for feature extraction)

    Logistic Regression

🧪 Sample Output

url,title,category,predicted_category
https://www.amazon.in/product123,Buy Now,product,product
https://blog.example.com/tips,Top 10 Tips,other,other

❓ Having Issues?

    Make sure you're using Python 3.10

    Run from the project root folder

    All scripts require the project structure to be preserved

