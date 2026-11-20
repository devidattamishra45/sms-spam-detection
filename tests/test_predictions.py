"""
End-to-End Prediction Quality Tests for SMS Spam & Phishing Detection System
"""

import pytest
from predict import predict_message


class TestPredictionQuality:
    """Verifies that the trained model accurately classifies standard real-world SMS examples."""

    @pytest.mark.parametrize(
        "spam_text",
        [
            "WINNER!! As a valued network customer you have won a £900 prize reward! Call 09061701461 to claim.",
            "URGENT! Your banking access has been restricted. Visit https://secure-bank-login-update.com immediately to verify your KYC.",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate).",
            "Congratulations! You have been selected for a $1,000 Walmart gift card. Click http://scam-rewards.org/gift to redeem now."
        ]
    )
    def test_spam_detection(self, spam_text: str):
        """Standard spam and phishing messages must be flagged as Spam."""
        res = predict_message(spam_text)
        assert res["is_valid"] is True
        assert res["label"] == "Spam"
        assert res["is_spam"] is True
        assert res["spam_probability_uncalibrated"] > 0.5

    @pytest.mark.parametrize(
        "ham_text",
        [
            "Hey mom, I'll be home late tonight. Can you save some dinner for me?",
            "Hi team, reminder that our project status meeting is at 3:00 PM in Conference Room B.",
            "Are you free to catch a coffee this Saturday afternoon? Let me know!",
            "Your appointment with Dr. Sharma has been confirmed for tomorrow at 10:30 AM."
        ]
    )
    def test_legitimate_detection(self, ham_text: str):
        """Everyday legitimate messages must be classified as Legitimate."""
        res = predict_message(ham_text)
        assert res["is_valid"] is True
        assert res["label"] == "Legitimate"
        assert res["is_spam"] is False
        assert res["ham_probability_uncalibrated"] > 0.5
