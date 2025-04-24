# %%
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Load documents assuming all are from 'aydın'
def load_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        author_files = folder_path + "/" + file_name
        for txt_name in os.listdir(author_files):
            if txt_name.endswith(".txt"):
                with open(
                    os.path.join(author_files, txt_name), "r", encoding="utf-8"
                ) as f:
                    text = f.read()
                    documents.append((text, file_name))
    return documents


folder_path = "datas/"
data = load_documents_from_folder(folder_path)

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "author"])
le = LabelEncoder()
df["label"] = le.fit_transform(df["author"])

# Her yazının kelime sayısını hesapla
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
author_word_counts = (
    df.groupby("author")["word_count"].sum().sort_values(ascending=False)
)
print("Her yazarın toplam kelime sayısı:")
print(df)


X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)


# %%
import re


def clean_text(text):
    text = re.sub(r"\*+", " ", text)  # *** -> boşluk
    text = re.sub(r"\n+", "\n", text)  # çoklu satır boşluklarını teke indir
    text = re.sub(r"[“”]", '"', text)  # fancy tırnakları düzleştir
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\s{2,}", " ", text)  # fazla boşluğu tek boşluk yap
    return text.strip()


df["text"] = df["text"].apply(clean_text)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
author_word_counts = (
    df.groupby("author")["word_count"].sum().sort_values(ascending=False)
)
print("Her yazarın toplam kelime sayısı:")
print(author_word_counts)


# %%
# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "Naive Bayes": MultinomialNB(),
    "MLP": MLPClassifier(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
}

# Define TF-IDF vectorizer settings
vectorizer_settings = {
    "word_unigram": TfidfVectorizer(analyzer="word", ngram_range=(1, 1)),
    "word_bigram_trigram": TfidfVectorizer(analyzer="word", ngram_range=(2, 3)),
    "char_bigram_trigram": TfidfVectorizer(analyzer="char", ngram_range=(2, 3)),
}

# %%
results = []

# TF-IDF based evaluations
for vec_name, vectorizer in vectorizer_settings.items():
    for model_name, model in models.items():
        print(f"Evaluating {model_name} with {vec_name}...")
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        results.append(
            {
                "Feature": vec_name,
                "Model": model_name,
                "Accuracy": report["accuracy"],
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-score": report["weighted avg"]["f1-score"],
            }
        )

print("TF-IDF Results:")
for result in results:
    print(
        f"Feature: {result['Feature']}, Model: {result['Model']}, "
        f"Accuracy: {result['Accuracy']:.4f}, Precision: {result['Precision']:.4f}, "
        f"Recall: {result['Recall']:.4f}, F1-score: {result['F1-score']:.4f}"
    )

# %%
# BERT embeddings
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
X_train_bert = bert_model.encode(X_train.tolist(), show_progress_bar=True)
X_test_bert = bert_model.encode(X_test.tolist(), show_progress_bar=True)

bert_compatible_models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "MLP": MLPClassifier(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
}

# TMP
results = []

for model_name, model in bert_compatible_models.items():
    print(f"Evaluating {model_name} with BERT embeddings...")
    model.fit(X_train_bert, y_train)
    y_pred = model.predict(X_test_bert)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results.append(
        {
            "Feature": "BERT",
            "Model": model_name,
            "Accuracy": report["accuracy"],
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-score": report["weighted avg"]["f1-score"],
        }
    )

print("BERT Results:")
for result in results:
    if result["Feature"] == "BERT":
        print(
            f"Feature: {result['Feature']}, Model: {result['Model']}, "
            f"Accuracy: {result['Accuracy']:.4f}, Precision: {result['Precision']:.4f}, "
            f"Recall: {result['Recall']:.4f}, F1-score: {result['F1-score']:.4f}"
        )
# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results2.csv", index=False)
print("Results saved to results.csv")
