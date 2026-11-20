"""
SMS Spam & Phishing Message Detection - Training Pipeline
=========================================================

Production-grade Machine Learning pipeline for telecommunications security.

This script implements the end-to-end Machine Learning lifecycle:
1. Dataset Ingestion:
   - Downloads official UCI SMS Spam Collection (ID: 228) using `ucimlrepo`.
   - Robust fallback to direct UCI archive download if package API shifts.
   - Saves local copy to `data/sms_spam_collection.csv` preserving original labels ('ham', 'spam').
2. Data Preprocessing:
   - Cleans and normalizes messages using `preprocess.clean_text`.
   - Preserves phishing anchors (URLs, currency symbols, hotline numbers).
3. Feature Extraction (TF-IDF):
   - Computes Term Frequency-Inverse Document Frequency (TF-IDF) representation.
   - Leverages unigrams and bigrams (ngram_range=(1, 2)) to capture contextual pairs
     (e.g., 'free cash', 'urgent call', 'click here').
4. Classification Algorithm (Multinomial Naive Bayes):
   - Combines TF-IDF and MultinomialNB into a Scikit-Learn Pipeline.
   - 80:20 Train/Test split using Stratified Sampling (stratify=y) and random_state=42.
   - Strictly fits TF-IDF ONLY on training data to prevent Data Leakage.
5. Model Evaluation:
   - Computes Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
   - Produces a detailed Classification Report.
   - Explains False Positives vs. False Negatives in telecom contexts.
6. Serialization:
   - Serializes trained Pipeline to `models/spam_detector_pipeline.joblib`.
   - Saves metrics to `models/metrics.json` for the Streamlit dashboard.
"""

import os
import sys
import json
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

from preprocess import clean_text

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_CSV_PATH = os.path.join(DATA_DIR, "sms_spam_collection.csv")
MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "spam_detector_pipeline.joblib")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_uci_dataset() -> pd.DataFrame:
    """
    Downloads the official UCI SMS Spam Collection dataset using ucimlrepo
    with automatic fallback to direct UCI archive download.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'label' ('ham' or 'spam') and 'message' (SMS text).
    """
    # 1. Check if local copy already exists to avoid redundant downloads
    if os.path.exists(DATASET_CSV_PATH):
        print(f"[*] Found local copy of dataset at: {DATASET_CSV_PATH}")
        df = pd.read_csv(DATASET_CSV_PATH)
        if "label" in df.columns and "message" in df.columns:
            return df

    print("[*] Downloading UCI SMS Spam Collection dataset (UCI ID: 228)...")

    # 2. Attempt download via ucimlrepo package
    try:
        from ucimlrepo import fetch_ucirepo
        print("  -> Attempting download via 'ucimlrepo' Python package...")
        sms_spam = fetch_ucirepo(id=228)
        
        # Extract features and targets
        X = sms_spam.data.features
        y = sms_spam.data.targets
        
        # Combine into standardized DataFrame
        df = pd.DataFrame()
        # Handle variations in ucimlrepo column naming
        if isinstance(X, pd.DataFrame):
            df["message"] = X.iloc[:, 0]
        else:
            df["message"] = pd.Series(X)
            
        if isinstance(y, pd.DataFrame):
            df["label"] = y.iloc[:, 0]
        else:
            df["label"] = pd.Series(y)

        # Standardize labels to lowercase 'ham' and 'spam'
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["message"] = df["message"].astype(str)

        # Cache locally
        df.to_csv(DATASET_CSV_PATH, index=False)
        print(f"[+] Successfully fetched via ucimlrepo and saved to {DATASET_CSV_PATH}")
        return df

    except Exception as e:
        print(f"  [!] ucimlrepo download encountered: {e}")
        print("  -> Falling back to direct download from UCI Machine Learning Repository...")

    # 3. Direct HTTP fallback to UCI repository zip archive
    uci_zip_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    try:
        response = requests.get(uci_zip_url, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The archive contains 'SMSSpamCollection'
            filename = [name for name in z.namelist() if "SMSSpamCollection" in name][0]
            with z.open(filename) as f:
                # Tab-separated file: label \t message
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=["label", "message"],
                    encoding="utf-8",
                    on_bad_lines="skip"
                )

        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["message"] = df["message"].astype(str)

        # Cache locally
        df.to_csv(DATASET_CSV_PATH, index=False)
        print(f"[+] Successfully downloaded via direct UCI URL and saved to {DATASET_CSV_PATH}")
        return df

    except Exception as e:
        print(f"  [!] Direct UCI download also failed: {e}")
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "DATASET DOWNLOAD FAILED!\n"
            "Please ensure you have an active internet connection.\n"
            "Manual Setup Instructions:\n"
            "1. Visit https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
            "2. Download 'sms+spam+collection.zip'\n"
            "3. Extract 'SMSSpamCollection'\n"
            f"4. Save it as a CSV at: {DATASET_CSV_PATH} with columns ['label', 'message']\n"
            "=" * 60
        )


def build_pipeline() -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn Pipeline combining TF-IDF feature
    extraction and a Multinomial Naive Bayes classifier.

    Why Scikit-learn Pipeline?
    --------------------------
    A Pipeline ensures that the exact same transformations applied to training data
    are seamlessly applied to unseen test/production data. Crucially, it completely
    prevents 'Data Leakage' because fit() is only called on X_train. The vocabulary
    and inverse document frequencies are learned strictly from the training set,
    and X_test is only transformed.

    Why TF-IDF (Term Frequency - Inverse Document Frequency)?
    ---------------------------------------------------------
    Computers cannot perform classification directly on strings; they require a numerical
    vector space.
    - Term Frequency (TF): Measures how frequently word 't' appears in SMS 'd'.
    - Inverse Document Frequency (IDF): Down-weights common words (e.g., 'to', 'is', 'the')
      that occur across nearly every document, while boosting discriminative terms
      (e.g., 'prize', 'urgent', 'claim', 'guaranteed', 'url_token').
    - Unigrams + Bigrams (ngram_range=(1, 2)): Captures both single words ('free')
      and word pairs ('free cash', 'call now', 'urgent update') which provide vital
      contextual signal for spam detection.

    Why Multinomial Naive Bayes (MultinomialNB)?
    -------------------------------------------
    MultinomialNB is exceptionally fast, highly effective for high-dimensional sparse
    text data, and requires minimal training samples. It calculates the posterior
    probability P(Spam | Words) using Bayes' Theorem:
        P(Class | Features) ∝ P(Class) * ∏ P(word_i | Class)
    """
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),       # Capture both single words and two-word phrases
                min_df=2,                 # Ignore ultra-rare typos appearing only once
                max_df=0.95,              # Ignore corpus-wide words appearing in >95% of texts
                sublinear_tf=True         # Apply sublinear scaling: 1 + log(TF) to dampen extreme frequencies
            )
        ),
        (
            "classifier",
            MultinomialNB(
                alpha=0.1                 # Laplace/Lidstone smoothing parameter (avoids zero probabilities)
            )
        )
    ])
    return pipeline


def train_and_evaluate(
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Executes dataset loading, stratified splitting, model fitting, and evaluation.

    Parameters
    ----------
    test_size : float, default=0.20
        Fraction of dataset allocated to the holdout test set (80:20 split).
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        Trained pipeline and metrics dictionary.
    """
    print("\n" + "=" * 60)
    print(" SMS SPAM & PHISHING DETECTION MODEL TRAINING ")
    print("=" * 60)

    # 1. Ingest dataset
    df = download_uci_dataset()
    print(f"[*] Total dataset records: {len(df)}")
    class_counts = df["label"].value_counts().to_dict()
    print(f"[*] Class distribution: {class_counts}")

    # 2. Text Preprocessing
    print("[*] Preprocessing SMS messages (lowercasing, URL normalization, cleaning)...")
    # Clean text while preserving phishing markers
    df["cleaned_message"] = df["message"].apply(lambda msg: clean_text(msg, remove_stopwords=False))

    # Drop any edge-case empty cleaned records if any
    valid_mask = df["cleaned_message"].str.strip().str.len() > 0
    df = df[valid_mask].copy()

    X = df["cleaned_message"].values
    y = df["label"].values

    # 3. Stratified Train/Test Split (80:20)
    # Stratified sampling ensures the proportion of spam to legitimate messages
    # is identical in both the training set and the test set.
    print(f"[*] Splitting dataset: {int((1 - test_size) * 100)}% Train, {int(test_size * 100)}% Test (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"  -> Training samples: {len(X_train)} (Ham: {sum(y_train == 'ham')}, Spam: {sum(y_train == 'spam')})")
    print(f"  -> Testing samples:  {len(X_test)} (Ham: {sum(y_test == 'ham')}, Spam: {sum(y_test == 'spam')})")

    # 4. Initialize and Train Pipeline
    print("[*] Training TF-IDF + Multinomial Naive Bayes Pipeline (Preventing Data Leakage)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[+] Model training complete!")

    # 5. Evaluate on Holdout Test Set
    print("\n" + "-" * 60)
    print(" EVALUATION RESULTS ON UNSEEN TEST DATA ")
    print("-" * 60)
    y_pred = pipeline.predict(X_test)

    # Quantitative metrics (treating 'spam' as the positive class)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, pos_label="spam"))
    recall = float(recall_score(y_test, y_pred, pos_label="spam"))
    f1 = float(f1_score(y_test, y_pred, pos_label="spam"))
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist()
    # cm format: [[TN (Ham as Ham), FP (Ham as Spam)],
    #             [FN (Spam as Ham), TP (Spam as Spam)]]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).ravel()
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision (Spam): {precision * 100:.2f}%")
    print(f"Recall (Spam):    {recall * 100:.2f}%")
    print(f"F1-Score (Spam):  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Negatives  (Ham  classified as Ham):  {tn}")
    print(f"  False Positives (Ham  classified as Spam): {fp}")
    print(f"  False Negatives (Spam classified as Ham):  {fn}")
    print(f"  True Positives  (Spam classified as Spam): {tp}")
    print("\nClassification Report:\n", report_text)

    # 6. Deep Explanation of False Positives vs False Negatives
    print("-" * 60)
    print(" TELECOM IMPLICATIONS: FALSE POSITIVES vs FALSE NEGATIVES")
    print("-" * 60)
    print(
        "In SMS spam and phishing filtering for telecommunication networks and SMS gateways:\n"
        "1. FALSE POSITIVE (FP): A legitimate message is incorrectly flagged as SPAM.\n"
        "   - Consequence: A user misses an essential One-Time Password (OTP), bank alert,\n"
        "     flight booking, or emergency message. In telecommunications, high FPs lead to\n"
        "     extreme user frustration, lost business transactions, and regulatory penalties.\n"
        "   - Metric to optimize: PRECISION.\n\n"
        "2. FALSE NEGATIVE (FN): A fraudulent spam or phishing message slips into the inbox.\n"
        "   - Consequence: The subscriber might fall victim to credential harvesting,\n"
        "     malicious APK downloads, or financial fraud.\n"
        "   - Metric to optimize: RECALL.\n\n"
        "Trade-off Summary: Telecom filters typically demand near-zero False Positives (high Precision)\n"
        "while actively keeping False Negatives as low as safely possible (high Recall)."
    )

    # 7. Package metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm
        },
        "classification_report": report_dict,
        "dataset_stats": {
            "total_samples": len(df),
            "ham_count": class_counts.get("ham", 0),
            "spam_count": class_counts.get("spam", 0),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
    }

    # 8. Save Artifacts
    print("\n[*] Saving model pipeline and metrics...")
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Saved trained pipeline to: {MODEL_PIPELINE_PATH}")
    print(f"[+] Saved metrics JSON to:       {METRICS_JSON_PATH}")
    print("=" * 60)

    return pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
