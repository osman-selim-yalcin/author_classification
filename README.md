# Author Classification with Text Mining and Machine Learning

This project classifies authors based on their text using both traditional and deep learning-based feature extraction techniques.

## Features
- TF-IDF (word, word n-gram, character n-gram)
- BERT embeddings
- Multiple classifiers: SVM, Random Forest, XGBoost, Naive Bayes, MLP, Decision Tree
- Metrics: Accuracy, Precision, Recall, F1-score

## How to Run

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run project
python main.ipynb  # or main.py
