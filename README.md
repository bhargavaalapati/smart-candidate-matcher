# Smart Candidate Matcher

## Project Overview

The Smart Candidate Matcher is a Streamlit-based application designed to help recruiters and hiring managers analyze resumes. By uploading a resume, the application uses a pre-trained machine learning model to predict the most likely job category for the candidate, along with a confidence score. It also extracts and displays key skills directly from the resume text.

## Features

- **Resume Upload:** Supports PDF, DOCX, and TXT file formats.
- **AI-Powered Category Prediction:** Predicts the job category (e.g., "Data Science," "Web Development") of a candidate's resume.
- **Confidence Scoring:** Provides a confidence percentage for the predicted category.
- **Skill Extraction:** Automatically identifies and lists technical and soft skills from the resume text.
- **Simple UI:** A clean, easy-to-use interface built with Streamlit.

## Installation

To run the application locally, follow these steps:

### Prerequisites

- Python 3.x

### Setup

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/bhargavaalapati/smart-candidate-matcher.git](https://github.com/bhargavaalapati/smart-candidate-matcher.git)
    cd smart-candidate-matcher
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

3.  **Upload a resume and view the results.**

## License

This project is licensed under the MIT License.
