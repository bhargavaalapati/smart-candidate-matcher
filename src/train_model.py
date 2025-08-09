import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from load_dataset import load_dataset  # use our improved loader

# -------------------------
# Config
# -------------------------
MODEL_VERSION = "v1.1.0"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = load_dataset(cleaned=True)
X = df["clean_resume"]
y = df["Category"]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Vectorization
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# Choose Model (you can switch)
# -------------------------
# Option 1: Logistic Regression
clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")

# Option 2: Uncomment for RandomForest
# clf = RandomForestClassifier(n_estimators=300, random_state=42)

# Train
clf.fit(X_train_vec, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Save Model & Vectorizer
# -------------------------
vectorizer_path = os.path.join(MODEL_DIR, f"vectorizer_{MODEL_VERSION}.pkl")
model_path = os.path.join(MODEL_DIR, f"classifier_{MODEL_VERSION}.pkl")

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(clf, model_path)

print(f"\n📂 Models saved as:\n - {vectorizer_path}\n - {model_path}")
