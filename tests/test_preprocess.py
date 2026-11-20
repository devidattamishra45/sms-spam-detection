"""
Unit Tests for Text Preprocessing Module (preprocess.py)
"""

import pytest
from preprocess import clean_text, tokenize, DEFAULT_STOPWORDS


class TestPreprocess:
    """Test suite for string normalization, regex filtering, and tokenization."""

    def test_lowercase_conversion(self):
        """Verify that uppercase letters are converted to lowercase."""
        raw = "URGENT WINNER CLAIM YOUR REWARD NOW"
        expected = "urgent winner claim your reward now"
        assert clean_text(raw) == expected

    def test_punctuation_removal(self):
        """Verify that unnecessary punctuation is stripped while words stay separated."""
        raw = "Hello! Are you coming? Yes, I am... maybe."
        cleaned = clean_text(raw)
        assert "!" not in cleaned
        assert "?" not in cleaned
        assert "," not in cleaned
        assert "." not in cleaned
        assert "hello are you coming yes i am maybe" == cleaned

    def test_phishing_url_tokenization(self):
        """Verify that URLs are replaced by url_token when preserve_phishing_tokens is True."""
        raw = "Click here to claim: https://secure-bank-login.com/auth or http://bit.ly/scam"
        cleaned = clean_text(raw, preserve_phishing_tokens=True)
        assert "url_token" in cleaned
        assert "https://" not in cleaned
        assert "http://" not in cleaned

    def test_currency_symbol_tokenization(self):
        """Verify that currency symbols are mapped to money_token."""
        raw = "You won £500 or $1000 or €250 cash prize!"
        cleaned = clean_text(raw, preserve_phishing_tokens=True)
        assert "money_token" in cleaned
        assert "£" not in cleaned
        assert "$" not in cleaned
        assert "€" not in cleaned

    def test_phone_number_tokenization(self):
        """Verify that long phone/claim digit strings are replaced with numeric_token."""
        raw = "Call 09061701461 to claim your gift"
        cleaned = clean_text(raw, preserve_phishing_tokens=True)
        assert "numeric_token" in cleaned

    def test_stopword_removal(self):
        """Verify stopword filtering when remove_stopwords is enabled."""
        raw = "this is a test message for you and me"
        cleaned_with_stopwords = clean_text(raw, remove_stopwords=False)
        cleaned_no_stopwords = clean_text(raw, remove_stopwords=True)

        assert "this" in cleaned_with_stopwords
        assert "test message" in cleaned_no_stopwords
        assert "this" not in cleaned_no_stopwords
        assert "for" not in cleaned_no_stopwords
        assert "and" not in cleaned_no_stopwords

    def test_empty_and_none_input(self):
        """Verify handling of empty strings, whitespace, and None."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(None) == ""

    def test_non_string_input(self):
        """Verify graceful type conversion for numbers or non-string types."""
        assert clean_text(123456) == "numeric_token"

    def test_tokenize(self):
        """Verify tokenization splits words cleanly."""
        text = "winner claim cash money_token"
        tokens = tokenize(text)
        assert tokens == ["winner", "claim", "cash", "money_token"]
        assert tokenize("") == []
