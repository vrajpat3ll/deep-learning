import re
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    text = s.lower()  # lowercase

    # remove any links mentioned in the text
    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\S+@\S+', '', text)  # remove any emails
    text = re.sub(r'<.*?>', '', text)  # remove any html tags

    # remove any non alphabet characters, unnecessary for capturing meaning of sentences
    text = re.sub(r'[^a-z0-9?!\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespaces

    return text.strip()


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
