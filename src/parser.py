import os
import re
import fitz  # PyMuPDF for PDF
import docx2txt  # pip install docx2txt
from tabulate import tabulate
import string


# -------------------------
# File Type Parsers
# -------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(docx_path):
    """Extract text from DOCX."""
    return docx2txt.process(docx_path) or ""


def extract_text_from_txt(txt_path):
    """Extract text from TXT."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -------------------------
# Preprocessing Helpers
# -------------------------
def clean_text(text):
    """Lowercase, remove punctuation & extra spaces."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    return text.translate(str.maketrans("", "", string.punctuation))


def extract_name(text):
    """Attempt to detect candidate name using heuristics."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Heuristic: Name is usually in first 3 lines and contains letters only
    for line in lines[:3]:
        if re.match(r"^[A-Za-z\s]+$", line) and 2 <= len(line.split()) <= 4:
            return line.title()
    return "Unknown"


# -------------------------
# Skill Extraction
# -------------------------
def extract_skills(text):
    """Extract skills from text using synonyms and variations."""
    skills_dict = {
        "python": ["python", "py"],
        "javascript": ["javascript", "js"],
        "html": ["html", "hypertext markup language"],
        "css": ["css", "cascading style sheets"],
        "react": ["react", "react.js", "reactjs"],
        "node.js": ["node.js", "nodejs", "node"],
        "machine learning": ["machine learning", "ml"],
        "ai": ["ai", "artificial intelligence"],
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "sql": ["sql", "structured query language"],
        "aws": ["aws", "amazon web services"],
    }

    found = []
    text_clean = clean_text(text)

    for skill, variants in skills_dict.items():
        if any(v.lower() in text_clean for v in variants):
            found.append(skill.title())

    return sorted(set(found))


# -------------------------
# Main Runner
# -------------------------
if __name__ == "__main__":
    resumes_dir = "data/resumes"
    results = []

    for file in os.listdir(resumes_dir):
        file_path = os.path.join(resumes_dir, file)
        ext = file.lower()

        if ext.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        elif ext.endswith(".docx"):
            raw_text = extract_text_from_docx(file_path)
        elif ext.endswith(".txt"):
            raw_text = extract_text_from_txt(file_path)
        else:
            continue  # skip unsupported files

        name = extract_name(raw_text)
        skills = extract_skills(raw_text)
        results.append(
            [file, name, ", ".join(skills) if skills else "No skills detected"]
        )

    print(
        tabulate(
            results, headers=["File", "Candidate Name", "Skills Found"], tablefmt="grid"
        )
    )
