"""
Text Preprocessing Module for SMS Spam & Phishing Message Detection
===================================================================

Why is Text Preprocessing Needed in Machine Learning?
-----------------------------------------------------
1. Vocabulary Reduction & Dimensionality Control:
   Natural language contains huge lexical variability. Without preprocessing,
   "Win!", "win.", "WIN", and "winning" would be treated as completely distinct
   tokens. By converting text to lowercase and stripping arbitrary punctuation,
   we consolidate identical semantic meanings into a single feature representation,
   drastically reducing feature matrix sparsity.

2. Noise Elimination:
   Extraneous characters, trailing spaces, HTML entities, and formatting artifacts
   add noise to the statistical distribution of words. Cleaning ensures our
   algorithms focus on predictive content rather than formatting idiosyncrasies.

3. Domain-Specific Signal Preservation:
   In SMS spam and phishing attacks, fraudsters frequently utilize specific
   anchors: URLs (phishing links), currency symbols (£, $, €) indicating lottery
   or financial scams, and urgency words. Standardizing these into consistent tokens
   (e.g., replacing URLs with 'url_token') retains the critical behavioral signal
   without blowing up vocabulary size.

4. Enabling Numerical Feature Extraction:
   Computers and statistical algorithms cannot perform matrix operations directly
   on raw strings. Preprocessing converts noisy textual inputs into standardized,
   tokenizable strings that vectorizers (like TF-IDF) can transform into numerical vectors.
"""

import re
import string
from typing import List, Optional

# Standard English Stopwords list (self-contained to prevent external download dependencies)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}

# Regex patterns for text cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_OR_DIGITS_PATTERN = re.compile(r"\b\d{5,}\b")  # Long digit sequences (codes, hotlines)
CURRENCY_PATTERN = re.compile(r"[$£€₹]")
MULTIPLE_WHITESPACE = re.compile(r"\s+")


def clean_text(
    text: Optional[str],
    remove_stopwords: bool = False,
    preserve_phishing_tokens: bool = True
) -> str:
    """
    Cleans raw SMS message text through normalization, punctuation stripping,
    and optional stopword removal.

    Parameters
    ----------
    text : Optional[str]
        Raw input string from SMS message.
    remove_stopwords : bool, default=False
        If True, filters out common English stopwords. By default False,
        as short SMS context can be preserved better without aggressive filtering.
    preserve_phishing_tokens : bool, default=True
        If True, replaces URLs, long digit strings (hotlines), and currency symbols
        with standardized semantic tokens ('url_token', 'numeric_token', 'money_token')
        so the classifier retains essential phishing cues without overfitting on specific URLs.

    Returns
    -------
    str
        Preprocessed, normalized, and cleaned text string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase conversion for case insensitivity
    cleaned = text.lower()

    # 2. Phishing-specific semantic tokens
    if preserve_phishing_tokens:
        # Normalize URLs (crucial for phishing link detection)
        cleaned = URL_PATTERN.sub(" url_token ", cleaned)
        # Normalize emails
        cleaned = EMAIL_PATTERN.sub(" email_token ", cleaned)
        # Normalize currency symbols (scams/lotteries frequently promise cash)
        cleaned = CURRENCY_PATTERN.sub(" money_token ", cleaned)
        # Normalize long phone / pin / claim numbers
        cleaned = PHONE_OR_DIGITS_PATTERN.sub(" numeric_token ", cleaned)
    else:
        # Strip URLs entirely if not preserving
        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = EMAIL_PATTERN.sub(" ", cleaned)

    # 3. Punctuation removal
    # Keep alphanumeric characters, underscores (used in tokens), and whitespace
    # Replace standard punctuation with spaces to avoid joining adjacent words
    punctuation_to_remove = string.punctuation.replace("_", "")  # preserve token underscores
    translator = str.maketrans(punctuation_to_remove, " " * len(punctuation_to_remove))
    cleaned = cleaned.translate(translator)

    # 4. Tokenization (whitespace-based split)
    tokens = cleaned.split()

    # 5. Optional Stopword filtering
    if remove_stopwords:
        tokens = [token for token in tokens if token not in DEFAULT_STOPWORDS]

    # 6. Re-join tokens with normalized single spacing
    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """
    Tokenizes preprocessed text into individual word tokens.

    Parameters
    ----------
    text : str
        Cleaned text string.

    Returns
    -------
    List[str]
        List of alphanumeric tokens.
    """
    if not text:
        return []
    return text.split()


if __name__ == "__main__":
    # Demonstration of preprocessing steps
    sample_spam = "URGENT! You have won £1,000 cash! Call 09061701461 now or claim at http://claim-prize.com to collect."
    sample_ham = "Hey mom, I will be home by 7:30 PM. Don't forget to buy milk!"

    print("--- Text Preprocessing Demonstration ---")
    print(f"Original Spam:\n  {sample_spam}")
    print(f"Cleaned Spam (with phishing tokens):\n  {clean_text(sample_spam)}")
    print(f"Cleaned Spam (with stopwords removed):\n  {clean_text(sample_spam, remove_stopwords=True)}")
    print(f"\nOriginal Legitimate:\n  {sample_ham}")
    print(f"Cleaned Legitimate:\n  {clean_text(sample_ham)}")
