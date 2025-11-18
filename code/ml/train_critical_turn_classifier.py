"""
Train a simple critical-turn classifier on synthetic turn-level data.

Pipeline:
- Load data/synthetic/study1_turns_labeled_synthetic.csv
- Filter to junior speakers
- TF-IDF on 'text'
- Logistic regression classifier
- Train/test split
- Save model + metrics

Outputs:
- models/critical_turn_classifier_synthetic.joblib
- fig/critical_turn_classifier_metrics.csv
- fig/critical_turn_classifier_metrics.txt
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)

    df = pd.read_csv("data/synthetic/study1_turns_labeled_synthetic.csv")
    df = df[df["is_junior"] == 1].copy()

    X_text = df["text"].astype(str).values
    y = df["is_critical"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    joblib.dump(
        dict(vectorizer=vectorizer, classifier=clf),
        "models/critical_turn_classifier_synthetic.joblib",
    )

    # Save metrics CSV
    metrics_df = pd.DataFrame(
        [
            dict(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                n_test=len(y_test),
            )
        ]
    )
    metrics_df.to_csv("fig/critical_turn_classifier_metrics.csv", index=False)

    # Save human-readable report
    report = classification_report(y_test, y_pred, digits=3)
    cm_str = np.array2string(cm)

    with open("fig/critical_turn_classifier_metrics.txt", "w") as f:
        f.write("Critical-turn classifier metrics (synthetic data)\n")
        f.write("===============================================\n\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"Precision (pos=critical): {prec:.3f}\n")
        f.write(f"Recall (pos=critical): {rec:.3f}\n")
        f.write(f"F1 (pos=critical): {f1:.3f}\n")
        f.write(f"n_test: {len(y_test)}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(cm_str)

    print("Saved model to models/critical_turn_classifier_synthetic.joblib")
    print("Saved metrics to fig/critical_turn_classifier_metrics.csv")
    print("Saved detailed metrics to fig/critical_turn_classifier_metrics.txt")


if __name__ == "__main__":
    main()
