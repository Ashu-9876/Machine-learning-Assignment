# Machine-learning-Assignment
# Job Classification and Recommendation System

This project implements a **Job Classification and Job Recommendation System** using deep learning and NLP techniques. It aims to accurately classify job descriptions into job titles and recommend similar job roles based on user input.

## Features

- **Job Title Classification**: Uses a CNN model to classify job descriptions into specific job titles.
- **Job Recommendation**: Suggests top 5 most relevant job titles using content-based filtering with TF-IDF and cosine similarity.
- **Evaluation Metrics**: Assessed using Precision, Recall, F1-Score, and Mean Average Precision (MAP).

---

##  Problem Statement

**Objective**:  
Build a system that can automatically classify job postings and provide recommendations based on textual input such as job descriptions or resumes.

- **Input**: Job description (text)
- **Output**: Predicted job title and top 5 similar job titles

---

## Tech Stack

- Python 🐍  
- TensorFlow & Keras 🧠 (for CNN model)
- NLTK (for text preprocessing)
- Scikit-learn (for evaluation & TF-IDF)
- Pandas & NumPy (for data handling)

---

## Libraries Used

- `tensorflow`, `keras`
- `nltk`
- `pandas`, `numpy`
- `sklearn` (LabelEncoder, TfidfVectorizer, cosine_similarity, classification_report)

---

## How It Works

1. **Data Preprocessing**:
   - Tokenization, stopword removal, lemmatization using NLTK
   - Text to sequence conversion using Keras Tokenizer

2. **Model Training**:
   - CNN model classifies the input job description into one of the job titles
   - Trained using `sparse_categorical_crossentropy`

3. **Job Recommendation**:
   - TF-IDF vectorization on preprocessed descriptions
   - Cosine similarity used to find the top 5 most similar jobs

4. **Evaluation**:
   - Generates classification report with precision, recall, F1-Score
   - Calculates MAP (Mean Average Precision)

---

## Example Use

```bash
Enter a job description or your resume summary:
"Design and implement data models and pipelines for analytics"

Output:
Predicted Job Title: Data Analyst

Top 5 Recommended Job Titles:
→ Data Analyst (Score: 0.276)
→ Business Intelligence Analyst (Score: 0.251)
→ Data Scientist (Score: 0.243)
→ Analytics Engineer (Score: 0.235)
→ BI Developer (Score: 0.229)
