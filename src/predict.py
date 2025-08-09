import os
import fitz  # PyMuPDF
import docx2txt
import joblib
from parser import clean_text, extract_skills  # reuse functions from parser.py

# -------------------------
# Config
# -------------------------
MODEL_VERSION = "v1.1.0"
MODEL_DIR = "models"

vectorizer = joblib.load(os.path.join(MODEL_DIR, f"vectorizer_{MODEL_VERSION}.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, f"classifier_{MODEL_VERSION}.pkl"))


# -------------------------
# File Readers
# -------------------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(path):
    return docx2txt.process(path) or ""


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -------------------------
# Prediction Pipeline
# -------------------------
def predict_resume_category(file_path):
    """Predict category and extract skills from a resume file."""
    ext = file_path.lower()

    if ext.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    elif ext.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)
    elif ext.endswith(".txt"):
        raw_text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Clean & process
    text_clean = clean_text(raw_text)
    skills = extract_skills(raw_text)  # raw for skill detection

    # Vectorize & predict
    X_vec = vectorizer.transform([text_clean])
    predicted_class = classifier.predict(X_vec)[0]
    predicted_proba = classifier.predict_proba(X_vec).max()  # confidence score

    return predicted_class, predicted_proba, skills


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    test_folder = "data/test_resumes"

    for file in os.listdir(test_folder):
        file_path = os.path.join(test_folder, file)
        try:
            category, confidence, skills = predict_resume_category(file_path)
            print(f"📄 {file}")
            print(f"   🏷 Category: {category} ({confidence*100:.2f}% confidence)")
            print(
                f"   💡 Skills Found: {', '.join(skills) if skills else 'No skills detected'}\n"
            )
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
