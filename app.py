"""
SMS Spam and Phishing Message Detection System
===============================================
Interactive Streamlit Application for SMS Spam and Phishing Detection.

Features:
- Real-time SMS Message Inference with instant classification (Spam vs. Legitimate).
- Clearly labeled uncalibrated probability estimates with theoretical rationale.
- Prominent Security Advisory warning against opening suspicious links.
- Interactive quick-load presets for classic phishing, spam, OTPs, and personal messages.
- Model Evaluation tab featuring Confusion Matrix Heatmap, Classification Report, and
  telecom-specific analysis of False Positives vs. False Negatives.
- Dataset Exploration tab with class distribution and searchable UCI SMS collection.
- NLP Architecture tab explaining text preprocessing, TF-IDF, N-grams, and Naive Bayes.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from predict import SpamDetector, DEFAULT_MODEL_PATH
from preprocess import clean_text

# Page Configuration
st.set_page_config(
    page_title="SMS Spam & Phishing Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern, telecom-grade presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .security-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 6px;
        color: #92400E;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .spam-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 2px solid #EF4444;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    .ham-badge {
        background-color: #DCFCE7;
        color: #166534;
        border: 2px solid #22C55E;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    .uncalibrated-note {
        font-size: 0.82rem;
        color: #6B7280;
        font-style: italic;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sms_spam_collection.csv")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")


@st.cache_resource
def load_detector():
    """Caches the trained model detector."""
    detector = SpamDetector(DEFAULT_MODEL_PATH)
    return detector


@st.cache_data
def load_metrics():
    """Loads saved training metrics."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset():
    """Loads the cached UCI SMS dataset."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return None


# Sidebar Navigation & Project Info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.markdown("### **System Control Panel**")
    st.markdown("**Domain:** Telecommunications Security & NLP")
    st.markdown("**Objective:** SMS & Phishing Defense")
    st.markdown("---")

    st.markdown("#### **System Specifications**")
    st.markdown("- **Model:** Multinomial Naive Bayes")
    st.markdown("- **Vectorization:** TF-IDF (Unigram + Bigram)")
    st.markdown("- **Dataset:** UCI SMS Spam Collection (5,572 SMS)")
    st.markdown("- **Data Leakage Prevention:** Scikit-Learn Pipeline")
    st.markdown("---")

    st.markdown("#### **Project Artifacts**")
    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    st.markdown(f"- Model Status: {'✅ Loaded' if model_exists else '⚠️ Train Model'}")
    dataset_exists = os.path.exists(DATA_PATH)
    st.markdown(f"- Dataset Status: {'✅ Cached' if dataset_exists else '⚠️ Not Found'}")

    st.markdown("---")
    st.caption("Developed with Python, Scikit-learn, and Streamlit.")


# Main Header
st.markdown('<div class="main-title">🛡️ SMS Spam and Phishing Message Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production-oriented NLP Classification Engine for Mobile & Telecom Subscriber Safety</div>',
    unsafe_allow_html=True
)

# Mandatory Security Advisory Callout
st.markdown("""
<div class="security-banner">
    ⚠️ <strong>IMPORTANT SECURITY ADVISORY:</strong>
    Machine Learning classifications are statistical probability estimates and are <strong>not 100% guaranteed</strong>.
    Even if an SMS is labeled as <strong>Legitimate (Ham)</strong>, users should <strong>never click unverified hyperlinks</strong>,
    install unknown APKs, or share confidential OTPs/banking credentials.
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live SMS Analyzer",
    "📊 Model Performance & Telecom Evaluation",
    "📁 Dataset Exploration",
    "🧠 NLP Architecture & Theory"
])

detector = load_detector()
metrics = load_metrics()
dataset_df = load_dataset()


# ==========================================
# TAB 1: LIVE SMS ANALYZER
# ==========================================
with tab1:
    st.subheader("Interactive SMS Message Classification")
    st.write(
        "Enter any raw SMS message below or select one of the curated real-world presets "
        "to test the NLP classification pipeline in real time."
    )

    # Preset Example Selector
    col_preset, col_clear = st.columns([4, 1])
    
    presets = {
        "Select a demo preset...": "",
        "🚨 Phishing Scam (Bank Account Suspension)": (
            "URGENT: Your State Bank account has been locked due to suspicious activity. "
            "Please verify your KYC immediately at https://sbi-secure-update.com/login to restore access."
        ),
        "🎰 Promotional Spam (Cash Prize Winner)": (
            "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! "
            "Call 09061701461 now or claim at http://prize-rewards.org to collect."
        ),
        "🔐 Critical Legitimate OTP (Transactional)": (
            "Your One-Time Password (OTP) for transaction of Rs. 4,500 at Flipkart is 839201. "
            "Valid for 10 minutes. Security Alert: Do not share OTP with anyone."
        ),
        "💬 Everyday Conversational SMS (Legitimate)": (
            "Hey Dev, are we still meeting for the project discussion today at 3:30 PM in the library?"
        )
    }

    with col_preset:
        selected_preset = st.selectbox(
            "Quick-Fill Example Messages:",
            options=list(presets.keys()),
            index=0
        )

    # Initialize session state for text input
    if "sms_input_text" not in st.session_state:
        st.session_state["sms_input_text"] = ""

    if selected_preset != "Select a demo preset...":
        st.session_state["sms_input_text"] = presets[selected_preset]

    # Input text area
    user_sms = st.text_area(
        "SMS Message Text:",
        value=st.session_state["sms_input_text"],
        height=120,
        placeholder="Type or paste an SMS message here (e.g., 'Congratulations! You won $1000...')."
    )

    col_btn_predict, col_btn_reset = st.columns([1, 5])
    with col_btn_predict:
        predict_clicked = st.button("🚀 Classify SMS", type="primary", use_container_width=True)

    if predict_clicked:
        if not user_sms.strip():
            st.error("⚠️ Invalid Input: Message is empty or contains only whitespace. Please enter valid SMS text.")
        elif not detector.is_model_loaded():
            st.error("⚠️ Model pipeline artifact not found. Please run `python train_model.py` first.")
        else:
            result = detector.predict(user_sms)

            st.markdown("---")
            st.subheader("Classification Outcome")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if result["is_spam"]:
                    st.markdown("""
                    <div class="spam-badge">
                        🚨 SPAM / PHISHING DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                    st.error(
                        "**Verdict:** High probability of fraudulent solicitation, phishing, or unsolicited spam."
                    )
                else:
                    st.markdown("""
                    <div class="ham-badge">
                        🛡️ LEGITIMATE (HAM) MESSAGE
                    </div>
                    """, unsafe_allow_html=True)
                    st.success(
                        "**Verdict:** Standard transactional, personal, or legitimate communication."
                    )

                st.markdown("#### **Preprocessing Inspection**")
                st.write("**Cleaned & Normalized Representation:**")
                st.code(result["cleaned_text"] if result["cleaned_text"] else "(Empty after cleaning)", language="text")

            with res_col2:
                st.markdown("#### **Model Probability Estimates**")
                spam_prob = result["spam_probability_uncalibrated"]
                ham_prob = result["ham_probability_uncalibrated"]

                st.write(f"**Spam Probability:** `{spam_prob * 100:.2f}%`")
                st.progress(float(spam_prob))

                st.write(f"**Legitimate (Ham) Probability:** `{ham_prob * 100:.2f}%`")
                st.progress(float(ham_prob))

                st.markdown("""
                <div class="uncalibrated-note">
                    📌 <strong>Notice: Probability Estimates are Uncalibrated.</strong><br>
                    Multinomial Naive Bayes applies the conditional independence assumption across all tokens.
                    Because words naturally correlate in text, posterior probabilities tend to push heavily toward
                    0.0 or 1.0. While the predicted classification label is highly dependable, these raw values
                    represent uncalibrated decision confidences rather than true empirical likelihoods.
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# TAB 2: MODEL EVALUATION & TELECOM EVALUATION
# ==========================================
with tab2:
    st.subheader("Model Evaluation & Telecom Trade-Off Analysis")
    st.write(
        "Evaluation results computed strictly on the **unseen holdout test set (20% stratified split, 1,114 messages)**. "
        "The TF-IDF vectorizer was fitted exclusively on the training set to prevent data leakage."
    )

    if metrics is None:
        st.warning("Metrics file `models/metrics.json` not found. Please execute `python train_model.py` to generate metrics.")
    else:
        # Key Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy'] * 100:.2f}%</div>
                <div class="metric-label">Overall Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['precision'] * 100:.2f}%</div>
                <div class="metric-label">Precision (Spam)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['recall'] * 100:.2f}%</div>
                <div class="metric-label">Recall (Spam)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['f1_score'] * 100:.2f}%</div>
                <div class="metric-label">F1-Score (Spam)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Visual Confusion Matrix and Telecom Analysis
        col_cm, col_telecom = st.columns([1, 1])

        with col_cm:
            st.markdown("#### **Confusion Matrix (Test Set)**")
            cm_data = np.array(metrics["confusion_matrix"]["matrix"])

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Pred: Ham", "Pred: Spam"],
                yticklabels=["Actual: Ham", "Actual: Spam"],
                cbar=False,
                ax=ax
            )
            plt.ylabel("Actual Label", fontsize=11)
            plt.xlabel("Predicted Label", fontsize=11)
            plt.title("Holdout Test Confusion Matrix", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close()

            cm_dict = metrics["confusion_matrix"]
            st.markdown(f"""
            - **True Negatives (TN):** {cm_dict['tn']} (Legitimate messages correctly passed)
            - **False Positives (FP):** {cm_dict['fp']} (Legitimate messages mistakenly flagged as spam)
            - **False Negatives (FN):** {cm_dict['fn']} (Spam messages missed)
            - **True Positives (TP):** {cm_dict['tp']} (Spam messages correctly blocked)
            """)

        with col_telecom:
            st.markdown("#### **Telecom & Carrier Implications: FP vs. FN**")
            st.markdown("""
            In carrier-level telecom networks and SMS gateway infrastructures,
            the trade-off between **False Positives** and **False Negatives** is the single
            most critical design decision:

            1. **False Positive (FP) — The Critical Penalty:**
               - *Definition:* A legitimate SMS is misclassified as Spam and dropped/quarantined.
               - *Impact:* Missing an OTP, two-factor authentication token, banking transaction alert,
                 railway ticket confirmation, or emergency broadcast.
               - *Telecom Consequence:* Severe customer outrage, broken enterprise SLAs, and regulatory scrutiny.
               - *Model Target:* **Maximize Precision on Spam** (our model achieves **99.29% Precision**,
                 with only 1 False Positive out of 965 legitimate test messages).

            2. **False Negative (FN) — The Security Threat:**
               - *Definition:* A fraudulent spam or phishing SMS bypasses the filter and reaches the subscriber.
               - *Impact:* Subscriber risk of phishing credential theft, malware installation, or financial fraud.
               - *Model Target:* **Maintain High Recall** (our model achieves **93.29% Recall** on Spam).
            """)

        st.markdown("---")
        st.markdown("#### **Full Classification Report**")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
        st.dataframe(report_df.style.format(precision=3), use_container_width=True)


# ==========================================
# TAB 3: DATASET EXPLORATION
# ==========================================
with tab3:
    st.subheader("Official UCI SMS Spam Collection Dataset Overview")
    st.write(
        "Source: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). "
        "The dataset contains 5,572 real SMS messages collected from mobile phone users and mobile forums."
    )

    if dataset_df is None:
        st.warning("Dataset not found locally. Run `python train_model.py` to download and cache.")
    else:
        col_d1, col_d2 = st.columns([1, 1])

        with col_d1:
            st.markdown("#### **Class Distribution**")
            class_counts = dataset_df["label"].value_counts()
            
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.pie(
                class_counts.values,
                labels=[f"Ham (Legitimate)\n{class_counts['ham']}", f"Spam\n{class_counts['spam']}"],
                autopct="%1.1f%%",
                colors=["#3B82F6", "#EF4444"],
                startangle=140,
                explode=(0, 0.1)
            )
            ax.axis("equal")
            plt.title("Class Balance (UCI SMS Spam Collection)", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close()

        with col_d2:
            st.markdown("#### **Dataset Key Statistics**")
            st.markdown(f"""
            - **Total SMS Messages:** `{len(dataset_df):,}`
            - **Legitimate Messages (Ham):** `{class_counts.get('ham', 0):,}` (`{class_counts.get('ham', 0)/len(dataset_df)*100:.1f}%`)
            - **Spam / Phishing Messages:** `{class_counts.get('spam', 0):,}` (`{class_counts.get('spam', 0)/len(dataset_df)*100:.1f}%`)
            - **Class Imbalance Ratio:** ~6.5 : 1 (Handled via Stratified Train/Test split)
            - **Train Split (80%):** 4,456 messages
            - **Test Split (20%):** 1,114 messages
            """)

        st.markdown("---")
        st.markdown("#### **Browse Raw Dataset Records**")
        
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            selected_class = st.radio("Filter by class:", ["All", "spam", "ham"], horizontal=True)
        with filter_col2:
            search_query = st.text_input("Search messages by keyword:", placeholder="e.g. prize, meeting, bank")

        filtered_df = dataset_df.copy()
        if selected_class != "All":
            filtered_df = filtered_df[filtered_df["label"] == selected_class]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df["message"].str.contains(search_query.strip(), case=False, na=False)]

        st.write(f"Showing **{len(filtered_df):,}** matching messages:")
        st.dataframe(filtered_df.head(50), use_container_width=True)


# ==========================================
# TAB 4: NLP ARCHITECTURE & THEORY
# ==========================================
with tab4:
    st.subheader("NLP Architecture, Mathematical Foundations & Interview Concepts")
    st.write(
        "Comprehensive architectural guide for NLP and Machine Learning engineering."
    )

    st.markdown("""
    ### 1. Why Text Preprocessing is Essential
    - **Computers cannot process raw text:** Natural language is unstructured, variable, and noisy.
      Mathematical models require structured numerical vectors.
    - **Vocabulary Pruning:** Without lowercasing and punctuation stripping, `"Prize"`, `"prize"`, and `"prize!"`
      would be allocated three distinct vector dimensions, diluting statistical strength.
    - **Domain-Specific Phishing Tokens:** Fraudulent SMS messages rely heavily on URLs, emergency hotline numbers,
      and monetary solicitations. Normalizing `https://...` to `url_token` and `£/$` to `money_token` ensures
      the model captures fraudulent intent rather than overfitting on individual web domains.

    ---

    ### 2. Feature Extraction: How TF-IDF Works
    **Term Frequency-Inverse Document Frequency (TF-IDF)** converts textual messages into a continuous numerical vector space:
    """)

    st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)")

    st.markdown("""
    - **Term Frequency $\\text{TF}(t, d)$:** Quantifies how frequently term $t$ occurs in SMS message $d$.
    - **Inverse Document Frequency $\\text{IDF}(t, D)$:** Downweights ubiquitous words (e.g. *"the"*, *"is"*, *"to"*)
      while amplifying rare, discriminative words (e.g. *"claim"*, *"guaranteed"*, *"urgent"*):
    """)

    st.latex(r"\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1")

    st.markdown("""
    - **Unigrams & Bigrams (`ngram_range=(1, 2)`):**
      Capturing word pairs is vital in telecom spam filtering. Single words like *"cash"* or *"free"* can appear
      in benign messages, but bigram phrases like **"free cash"**, **"urgent call"**, and **"click here"** provide
      unambiguous phishing cues.

    ---

    ### 3. Classification Algorithm: Multinomial Naive Bayes
    Multinomial Naive Bayes uses Bayes' theorem with the assumption of conditional feature independence:
    """)

    st.latex(r"P(\text{Class} \mid W) \propto P(\text{Class}) \prod_{i=1}^{n} P(w_i \mid \text{Class})")

    st.markdown("""
    - **Laplace Smoothing ($\alpha=0.1$):**
      Ensures that a previously unseen word in a test message does not zero out the entire product:
    """)

    st.latex(r"P(w_i \mid \text{Class}) = \frac{N_{c, i} + \alpha}{N_c + \alpha \times |V|}")

    st.markdown("""
    ---

    ### 4. Preventing Data Leakage with Scikit-Learn Pipeline
    **What is Data Leakage?**
    If TF-IDF vectorization is fitted on the entire dataset *before* the train/test split, the IDF weights
    and vocabulary are informed by the test set. This produces artificially optimistic test scores that fail in production.

    **Our Implementation:**
    We wrap `TfidfVectorizer` and `MultinomialNB` inside `sklearn.pipeline.Pipeline`.
    Calling `pipeline.fit(X_train, y_train)` fits vocabulary and IDFs exclusively on `X_train`.
    When evaluating `X_test`, the pipeline uses the pre-fitted parameters without looking at `X_test` statistics.
    """)
