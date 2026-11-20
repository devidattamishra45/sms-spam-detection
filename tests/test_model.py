"""
Unit & Integration Tests for Prediction Engine (predict.py) and Model Pipeline
"""

import os
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from predict import SpamDetector, predict_message
from train_model import build_pipeline


class TestSpamDetectorValidation:
    """Test suite for input sanitization and defensive edge-case handling."""

    def test_null_input(self):
        """None input must yield is_valid=False and informative error message."""
        detector = SpamDetector()
        res = detector.predict(None)
        assert res["is_valid"] is False
        assert res["label"] == "Invalid Input"
        assert "empty or null" in res["status_message"].lower()

    def test_empty_string(self):
        """Empty string must yield is_valid=False."""
        res = predict_message("")
        assert res["is_valid"] is False
        assert res["label"] == "Invalid Input"

    def test_whitespace_only_string(self):
        """Whitespace-only string must yield is_valid=False."""
        res = predict_message("   \t\n  ")
        assert res["is_valid"] is False
        assert res["label"] == "Invalid Input"
        assert "whitespace" in res["status_message"].lower()

    def test_warning_message_present(self):
        """Every prediction response must contain the security advisory."""
        res = predict_message("Valid test message")
        assert "warning" in res
        assert "security advisory" in res["warning"].lower()


class TestPipelineArchitecture:
    """Tests for pipeline construction, hyperparameters, and feature extraction."""

    def test_pipeline_components(self):
        """Verify pipeline contains TfidfVectorizer and MultinomialNB."""
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert "tfidf" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(pipeline.named_steps["classifier"], MultinomialNB)

    def test_tfidf_ngram_range(self):
        """Verify unigrams and bigrams are extracted (ngram_range=(1, 2))."""
        pipeline = build_pipeline()
        tfidf = pipeline.named_steps["tfidf"]
        assert tfidf.ngram_range == (1, 2)

    def test_pipeline_fit_and_predict(self):
        """Test micro-dataset training and prediction sanity check."""
        toy_corpus = [
            "claim free money win cash prize now",
            "urgent call winner guaranteed gift",
            "hi are we still meeting for lunch today",
            "please review the attached report when you have time",
            "can you pick up milk on the way home"
        ]
        toy_labels = ["spam", "spam", "ham", "ham", "ham"]

        pipeline = build_pipeline()
        pipeline.fit(toy_corpus, toy_labels)

        # Test prediction
        test_msg = ["congratulations win free prize cash"]
        pred = pipeline.predict(test_msg)[0]
        probs = pipeline.predict_proba(test_msg)[0]

        assert pred in ["spam", "ham"]
        assert len(probs) == 2
        assert 0.0 <= probs[0] <= 1.0
        assert 0.0 <= probs[1] <= 1.0
        assert pytest.approx(sum(probs), 0.001) == 1.0
