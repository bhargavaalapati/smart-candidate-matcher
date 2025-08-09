import pandas as pd
import re
from bs4 import BeautifulSoup
import spacy
import os

# Load English NLP model (for tokenization & lemmatization)
nlp = spacy.load(
    "en_core_web_sm"
)  # pip install spacy && python -m spacy download en_core_web_sm

# Skill dictionary with synonyms
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
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
}


def clean_text(text):
    """Remove HTML, punctuation, extra spaces, lowercase."""
    text = BeautifulSoup(str(text), "html.parser").get_text()  # strip HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # keep letters only
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text):
    """Lemmatize text to normalize word forms."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


def extract_skills(text):
    """Match skills using synonyms."""
    found = []
    for skill, variants in skills_dict.items():
        for variant in variants:
            if variant.lower() in text:
                found.append(skill)
                break  # avoid duplicates if multiple variants found
    return sorted(set(found))


if __name__ == "__main__":
    dataset_path = "data/resume_dataset/Resume/Resume.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Clean and preprocess
    df["clean_resume"] = df["Resume_str"].apply(clean_text).apply(lemmatize_text)
    df["skills_found"] = df["clean_resume"].apply(extract_skills)

    # Preview
    print(df[["Category", "skills_found"]].head())

    # Save processed dataset
    output_path = "data/resume_dataset/cleaned_resumes.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to {output_path}")
