# SMS Spam and Phishing Message Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-24%20Passed-brightgreen.svg)](https://pytest.org/)

An end-to-end, production-oriented Natural Language Processing (NLP) system designed to detect promotional spam, fraudulent schemes, and phishing attacks in mobile SMS traffic.

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Telecom Network Security Context](#2-telecom-network-security-context)
3. [Dataset Details & Acquisition](#3-dataset-details--acquisition)
4. [Text Preprocessing Architecture](#4-text-preprocessing-architecture)
5. [Feature Extraction (TF-IDF & N-Grams)](#5-feature-extraction-tf-idf--n-grams)
6. [Machine Learning Algorithm & Pipeline](#6-machine-learning-algorithm--pipeline)
7. [Model Evaluation & Telecom Trade-Offs](#7-model-evaluation--telecom-trade-offs)
8. [Interactive Streamlit Dashboard](#8-interactive-streamlit-dashboard)
9. [Project Structure](#9-project-structure)
10. [Setup & Execution Instructions](#10-setup--execution-instructions)
11. [Automated Testing Suite](#11-automated-testing-suite)
12. [Interview Q&A Cheatsheet](#12-interview-qa-cheatsheet)

---

## 1. Project Overview & Problem Statement

### The Problem
Mobile telephone subscribers receive high volumes of unwanted Short Message Service (SMS) traffic daily, ranging from unsolicited commercial promotions to aggressive financial scams and weaponized phishing links (smishing). These messages:
- Violate subscriber privacy and clutter device storage.
- Facilitate identity theft, credential harvesting, and financial fraud.
- Impose severe network congestion and reputational damage on telecommunications providers.

### The Objective
To build a high-precision, low-latency machine learning classification system that analyzes incoming SMS texts and reliably distinguishes between **Legitimate (Ham)** and **Spam/Phishing** communications before delivery to subscriber hand-held devices.

---

## 2. Telecom Network Security Context

In high-throughput telecommunications networks and carrier-level SMS gateways (SMSCs):
- **High Throughput & Millisecond Latency:** Complex deep neural networks (e.g. LLMs, heavy Transformers) introduce latency and significant computational cost per message. A **TF-IDF + Multinomial Naive Bayes pipeline** executes inference in under a millisecond, making it suitable for edge gateway filtering.
- **Ultra-Low False Positive Requirement:** In telecom switching, dropping a legitimate OTP, bank transfer confirmation, or government emergency alert is unacceptable. The model achieves **99.29% Precision** with only **1 False Positive out of 965 legitimate test messages**.

---

## 3. Dataset Details & Acquisition

The model is trained on the official **UCI SMS Spam Collection Dataset** (ID: 228).
- **Source:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Total Records:** 5,572 labeled SMS messages.
- **Distribution:**
  - **Legitimate (Ham):** 4,825 messages (86.6%)
  - **Spam:** 747 messages (13.4%)
  - **Class Imbalance:** ~6.5 : 1

### Automated Download & Fallback Strategy
`train_model.py` implements a resilient retrieval strategy:
1. Checks local cache at `data/sms_spam_collection.csv`.
2. Attempts download via the official `ucimlrepo` Python package.
3. Automatically falls back to the direct UCI Machine Learning archive HTTP endpoint (`https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`) and extracts `SMSSpamCollection`.
4. Saves a local CSV copy in `data/`, preserving original labels (`ham` and `spam`).

---

## 4. Text Preprocessing Architecture

Text preprocessing transforms variable human natural language into standardized, clean representations.

### Why Preprocessing is Essential
1. **Vocabulary Pruning:** Reduces feature space explosion. Without lowercasing and punctuation stripping, `"Winner"`, `"winner"`, and `"WINNER!"` would generate three independent dimensions.
2. **Noise Reduction:** Eliminates extraneous punctuation, multiple spaces, and typographic artifacts that add variance without predictive signal.
3. **Domain-Specific Phishing Tokens:** Rather than discarding URLs or phone numbers, our pipeline replaces them with standardized semantic tokens:
   - URLs $\to$ `url_token` (captures phishing link presence without overfitting to temporary domains)
   - Currency symbols (`£`, `$`, `€`, `₹`) $\to$ `money_token` (lottery and financial spam indicators)
   - Long phone/claim numbers $\to$ `numeric_token` (urgent callback scam indicators)
4. **Stopwords (Optional Experiment):** Stopwords (`is`, `the`, `and`) can be filtered, but in short SMS messages, retaining them can preserve crucial context. Configurable via `clean_text(remove_stopwords=False)`.

---

## 5. Feature Extraction (TF-IDF & N-Grams)

### Why Computers Need Numerical Features
Computers and machine learning algorithms perform linear algebra (vector dot products, matrix multiplications, probabilistic products). Raw strings cannot be directly processed by mathematical classifiers; they must be mapped into a vector space $\mathbb{R}^V$.

### TF-IDF Formulation
Term Frequency-Inverse Document Frequency balances word rarity across the corpus against frequency in individual messages:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- **Term Frequency $\text{TF}(t, d)$:** Measures how often term $t$ occurs in document $d$. In our pipeline, `sublinear_tf=True` applies logarithmic dampening: $1 + \log(\text{TF})$.
- **Inverse Document Frequency $\text{IDF}(t, D)$:** Penalizes omnipresent words and rewards discriminative keywords:

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Unigrams & Bigrams (`ngram_range=(1, 2)`)
Single words like *"free"* or *"call"* occasionally appear in legitimate personal messages. However, **bigrams (two-word phrases)** like `"free cash"`, `"claim now"`, `"urgent call"`, and `"click here"` carry unmistakable scam intent. Using `ngram_range=(1, 2)` gives the model contextual awareness without the combinatorial explosion of trigrams.

---

## 6. Machine Learning Algorithm & Pipeline

### Multinomial Naive Bayes (`MultinomialNB`)
Multinomial Naive Bayes models the probability distribution of count-based or frequency-based features under Bayes' Theorem:

$$P(\text{Class} \mid W) \propto P(\text{Class}) \prod_{i=1}^{n} P(w_i \mid \text{Class})$$

- **Laplace Smoothing ($\alpha=0.1$):**
  Guarantees that an unseen word in production does not assign zero posterior probability to the entire message:

$$P(w_i \mid \text{Class}) = \frac{N_{c, i} + \alpha}{N_c + \alpha \times |V|}$$

### Preventing Data Leakage with Scikit-Learn Pipeline
A common error in ML implementations is fitting a vectorizer on the full dataset before splitting. This allows test-set vocabulary and IDF statistics to leak into the training phase.
Our implementation uses `sklearn.pipeline.Pipeline`:
```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
    ("classifier", MultinomialNB(alpha=0.1))
])
# Strictly fits TF-IDF vocabulary and IDF weights on X_train only!
pipeline.fit(X_train, y_train)
```

---

## 7. Model Evaluation & Telecom Trade-Offs

The model was evaluated on a strict **20% holdout test set (1,114 messages)** using **Stratified Sampling (`stratify=y`)** with `random_state=42`.

### Evaluation Metrics Summary
| Metric | Value | Telecom Significance |
| :--- | :--- | :--- |
| **Accuracy** | **99.01%** | Overall correct classifications across all messages |
| **Precision (Spam)** | **99.29%** | When flagged as Spam, the probability it is genuinely spam |
| **Recall (Spam)** | **93.29%** | Percentage of all actual spam messages successfully intercepted |
| **F1-Score (Spam)** | **96.19%** | Harmonic balance between Precision and Recall |

### Holdout Confusion Matrix
$$\begin{pmatrix} 
\text{True Negatives (TN) = 964} & \text{False Positives (FP) = 1} \\
\text{False Negatives (FN) = 10} & \text{True Positives (TP) = 139}
\end{pmatrix}$$

### Deep Dive: False Positives vs. False Negatives
- **False Positive (FP):** A legitimate SMS is misclassified as Spam and dropped.
  - *Real-world Consequence:* A user misses an emergency alert, bank OTP, or travel ticket. In telecom, this creates severe customer dissatisfaction, regulatory penalties, and broken SLAs.
  - *Optimization Focus:* High **Precision**. Our model achieved **99.29% Precision**, with only **1 FP out of 965 legitimate messages**.
- **False Negative (FN):** A fraudulent SMS slips through into the user's inbox.
  - *Real-world Consequence:* A subscriber could be phished or lose funds.
  - *Optimization Focus:* High **Recall**. Our model caught **139 out of 149 spam attacks (93.29% Recall)**.

---

## 8. Interactive Streamlit Dashboard

The project features a modern, four-tab Streamlit dashboard (`app.py`):
1. **Live SMS Analyzer:**
   - Real-time classification with visual badges (Red 🚨 for Spam, Green 🛡️ for Legitimate).
   - Quick-load demo presets (Phishing scam, lottery spam, bank OTP, lunch chat).
   - **Uncalibrated Probability Display:** Probabilities are explicitly marked as *uncalibrated* with educational tooltips explaining the Naive Bayes independence assumption.
   - **Prominent Security Advisory:** Explicit warning that machine learning predictions are probabilistic and users must never open suspicious hyperlinks simply because a message was labeled legitimate.
2. **Model Evaluation Tab:** Interactive Confusion Matrix heatmap, metric cards, and full classification report.
3. **Dataset Exploration Tab:** Class distribution pie chart and searchable dataset browser.
4. **NLP Architecture Tab:** Mathematical explanations of preprocessing, TF-IDF, N-grams, and Naive Bayes for interview walkthroughs.

---

## 9. Project Structure

```
sms-spam-detection/
├── app.py                     # Streamlit web dashboard
├── train_model.py             # Dataset retrieval, training pipeline, and evaluation
├── preprocess.py              # Text cleaning, normalization, and tokenization module
├── predict.py                 # Inference engine with defensive input validation
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
├── README.md                  # Complete technical documentation & interview guide
├── data/
│   └── sms_spam_collection.csv # Local cached copy of UCI SMS dataset (5,572 rows)
├── models/
│   ├── spam_detector_pipeline.joblib # Serialized Scikit-learn Pipeline
│   └── metrics.json           # Evaluation metrics and confusion matrix
└── tests/
    ├── __init__.py
    ├── test_preprocess.py     # 9 unit tests for text cleaning and tokenization
    ├── test_model.py          # 7 unit tests for inference and pipeline validation
    └── test_predictions.py    # 8 end-to-end classification quality tests
```

---

## 10. Setup & Execution Instructions

### Prerequisites
- Python 3.9 or higher
- Git

### Step 1: Clone or Navigate to the Workspace
```bash
cd sms-spam-detection
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Automated Unit & Integration Tests
```bash
pytest tests/ -v
```
*(All 24 test cases will pass in ~1.2 seconds).*

### Step 5: Train the Model (Optional — Pre-trained Model Included)
```bash
python train_model.py
```
*(Fetches dataset, trains the pipeline, logs metrics, and updates `models/`).*

### Step 6: Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 11. Automated Testing Suite

The project includes 24 automated tests across three test modules:
1. `tests/test_preprocess.py`: Tests lowercasing, punctuation stripping, URL tokenization, currency tokenization, stopword filtering, and non-string inputs.
2. `tests/test_model.py`: Validates input defense (null inputs, empty strings, whitespace-only messages), pipeline construction, N-gram parameters, and security advisory inclusion.
3. `tests/test_predictions.py`: Validates end-to-end model behavior on real-world phishing and legitimate SMS messages.

Run tests anytime with:
```bash
pytest
```

---

## 12. Interview Q&A Cheatsheet

**Q1: Why did you choose Multinomial Naive Bayes instead of a Deep Learning model (like BERT or LSTM)?**
> *Answer:* In telecommunications routing (SMSCs), message volume reaches tens of thousands of SMS per second. Deep learning models introduce significant latency (50-200ms) and require expensive GPU infrastructure. Multinomial Naive Bayes with TF-IDF provides sub-millisecond inference (<1ms), uses negligible memory (<10MB), and achieves **99.01% accuracy and 99.29% precision**, satisfying carrier-grade throughput and reliability requirements.

**Q2: What is data leakage, and how did you prevent it?**
> *Answer:* Data leakage occurs when test data information inadvertently influences the model training process. If we fit a TF-IDF vectorizer across the entire dataset before splitting, the learned vocabulary and inverse document frequency (IDF) weights reflect the test set, creating overly optimistic metrics. I prevented this by wrapping feature extraction and the classifier inside a `sklearn.pipeline.Pipeline`, ensuring `fit()` is called exclusively on `X_train`.

**Q3: Why are the predicted probabilities labeled as "uncalibrated"?**
> *Answer:* Multinomial Naive Bayes assumes that features (words and bigrams) are conditionally independent given the class. In natural language, words frequently co-occur (e.g., "click" and "here"). This violated assumption does not hurt classification decisions, but it causes the posterior probability $P(\text{Spam} \mid W)$ to become overly confident—pushing outputs close to 0.0 or 1.0. Therefore, probabilities must be presented as uncalibrated decision scores rather than true empirical probabilities unless calibrated with Platt scaling or isotonic regression.

**Q4: Which is worse in telecom SMS spam filtering: a False Positive or a False Negative?**
> *Answer:* A False Positive is generally far more damaging. If a legitimate bank OTP, train reservation, or emergency communication is quarantined as spam, the subscriber suffers immediate disruption and the carrier faces SLA violations and regulatory fines. Therefore, we prioritize high Precision (99.29%) so legitimate messages are virtually never blocked, while maintaining strong Recall (93.29%) to catch phishing campaigns.

---

## License
Educational & Research Demonstration — Machine Learning Project.
