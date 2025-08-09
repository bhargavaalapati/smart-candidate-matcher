import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import joblib
import os
from src.parser import clean_text, extract_skills  # Reuse from parser.py

# -------------------------
# Config
# -------------------------
MODEL_VERSION = "v1.1.0"
MODEL_DIR = "models"

vectorizer = joblib.load(os.path.join(MODEL_DIR, f"vectorizer_{MODEL_VERSION}.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, f"classifier_{MODEL_VERSION}.pkl"))


# -------------------------
# File Extractors
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_docx(file):
    return docx2txt.process(file) or ""


def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart Candidate Matcher", page_icon="📄")
st.title("📄 Smart Candidate Matcher")
st.write(
    "Upload your resume and get the predicted **job category**, **confidence score**, and **skills detected**."
)

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_type = uploaded_file.name.lower()

    # Extract text based on file type
    if file_type.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    elif file_type.endswith(".docx"):
        raw_text = extract_text_from_docx(uploaded_file)
    elif file_type.endswith(".txt"):
        raw_text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    if not raw_text.strip():
        st.error("No readable text found in file.")
        st.stop()

    # Clean text for model
    text_clean = clean_text(raw_text)

    # Predict category + confidence
    X_vec = vectorizer.transform([text_clean])
    predicted_category = classifier.predict(X_vec)[0]
    confidence = classifier.predict_proba(X_vec).max()

    # Extract skills from raw text
    skills_found = extract_skills(raw_text)

    # -------------------------
    # Results Display
    # -------------------------
    st.subheader("Prediction Results")
    st.metric(label="Predicted Category", value=predicted_category)
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

    st.subheader("Skills Detected")
    if skills_found:
        st.write(", ".join(skills_found))
    else:
        st.write("No skills detected.")

    # Optional: Show extracted text
    with st.expander("📜 View Extracted Resume Text"):
        st.write(raw_text)
