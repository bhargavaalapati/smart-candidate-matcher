import pandas as pd
import os


def load_dataset(path="data/resume_dataset/Resume/Resume.csv", cleaned=False):
    """
    Load resume dataset from CSV.

    Args:
        path (str): Path to the dataset CSV.
        cleaned (bool): If True, load the cleaned dataset instead of raw.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if cleaned:
        path = "data/resume_dataset/cleaned_resumes.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Basic validation
    required_cols = (
        ["Resume_str", "Category"]
        if not cleaned
        else ["clean_resume", "skills_found", "Category"]
    )
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # Drop rows with missing resumes
    df.dropna(subset=[required_cols[0]], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Loaded dataset: {len(df)} resumes ({'cleaned' if cleaned else 'raw'})")
    return df


if __name__ == "__main__":
    # Example: Load raw dataset
    df_raw = load_dataset()
    print(df_raw.head(3))

    # Example: Load cleaned dataset
    df_clean = load_dataset(cleaned=True)
    print(df_clean.head(3))
