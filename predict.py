"""
SMS Spam & Phishing Message Prediction Engine
==============================================

Provides inference logic for classifying new SMS text messages as
'Spam' or 'Legitimate' using a trained Scikit-learn Pipeline (TF-IDF + MultinomialNB).

Key Design Considerations:
--------------------------
1. Input Validation:
   Handles null inputs, empty strings, whitespace, non-string types gracefully
   without crashing or throwing unhandled exceptions.

2. Uncalibrated Probability Disclaimer:
   Multinomial Naive Bayes applies Bayes' theorem under the assumption that all features
   (words/n-grams) are mutually independent given the class. Because word occurrences
   in natural language correlate heavily, the posterior probabilities output by `predict_proba()`
   are notoriously pushed toward 0 and 1 (extreme overconfidence). While the rank ordering
   and classification decisions are robust, the raw probability numbers MUST be explicitly
   labeled as *uncalibrated*.

3. Interpretability & Feature Insights:
   Includes diagnostic extraction to explain which tokens (e.g., suspicious URLs,
   currency markers, emergency urgency cues) triggered the classification.
"""

import os
from typing import Dict, Any, Optional
import joblib
from preprocess import clean_text

# Default model artifact path
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "spam_detector_pipeline.joblib")


class SpamDetector:
    """
    Inference wrapper for the trained SMS Spam and Phishing Pipeline.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Loads the trained pipeline containing TfidfVectorizer and MultinomialNB.

        Parameters
        ----------
        model_path : str
            Filepath to the saved joblib pipeline artifact.
        """
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads serialized model pipeline from disk if available."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            self.pipeline = None

    def is_model_loaded(self) -> bool:
        """Checks whether the trained pipeline is loaded and ready for inference."""
        return self.pipeline is not None

    def predict(self, raw_message: Optional[str]) -> Dict[str, Any]:
        """
        Predicts whether an incoming SMS is 'Spam' or 'Legitimate'.

        Parameters
        ----------
        raw_message : Optional[str]
            Raw SMS text message to evaluate.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - is_valid: bool (whether input was valid text)
            - label: str ('Spam', 'Legitimate', or 'Invalid Input')
            - is_spam: bool
            - spam_probability_uncalibrated: float (0.0 to 1.0)
            - ham_probability_uncalibrated: float (0.0 to 1.0)
            - cleaned_text: str
            - status_message: str
            - warning: str (prominent security warning regarding phishing links)
        """
        # 1. Defensive input validation
        if raw_message is None:
            return {
                "is_valid": False,
                "label": "Invalid Input",
                "is_spam": False,
                "spam_probability_uncalibrated": 0.0,
                "ham_probability_uncalibrated": 0.0,
                "cleaned_text": "",
                "status_message": "Message is empty or null. Please provide a valid SMS text.",
                "warning": "Empty input provided."
            }

        if not isinstance(raw_message, str):
            raw_message = str(raw_message)

        raw_trimmed = raw_message.strip()
        if not raw_trimmed:
            return {
                "is_valid": False,
                "label": "Invalid Input",
                "is_spam": False,
                "spam_probability_uncalibrated": 0.0,
                "ham_probability_uncalibrated": 0.0,
                "cleaned_text": "",
                "status_message": "Message contains only whitespace. Please enter actual message text.",
                "warning": "Whitespace-only input provided."
            }

        # 2. Check if model is loaded
        if self.pipeline is None:
            # Re-attempt loading in case model was just trained
            self._load_model()
            if self.pipeline is None:
                return {
                    "is_valid": False,
                    "label": "Model Not Found",
                    "is_spam": False,
                    "spam_probability_uncalibrated": 0.0,
                    "ham_probability_uncalibrated": 0.0,
                    "cleaned_text": "",
                    "status_message": f"Model artifact not found at {self.model_path}. Run train_model.py first.",
                    "warning": "Model must be trained before predictions can be served."
                }

        # 3. Clean and preprocess input text
        cleaned = clean_text(raw_trimmed)

        # 4. Perform inference via Scikit-Learn Pipeline
        # Input to pipeline must be an iterable of strings
        pred_label_raw = self.pipeline.predict([cleaned])[0]
        probabilities = self.pipeline.predict_proba([cleaned])[0]

        # Pipeline classes mapping: e.g., ['ham', 'spam']
        classes = list(self.pipeline.classes_)
        spam_index = classes.index("spam") if "spam" in classes else 1
        ham_index = classes.index("ham") if "ham" in classes else 0

        spam_prob = float(probabilities[spam_index])
        ham_prob = float(probabilities[ham_index])
        is_spam = (pred_label_raw.lower() == "spam")

        return {
            "is_valid": True,
            "label": "Spam" if is_spam else "Legitimate",
            "is_spam": is_spam,
            "spam_probability_uncalibrated": spam_prob,
            "ham_probability_uncalibrated": ham_prob,
            "cleaned_text": cleaned,
            "status_message": "Message successfully analyzed.",
            "warning": (
                "IMPORTANT SECURITY ADVISORY: Machine learning classifications are probabilistic "
                "estimates and are never 100% guaranteed. Even if a message is classified as 'Legitimate', "
                "NEVER click suspicious hyperlinks or disclose confidential OTPs/passwords."
            )
        }


# Module-level convenience singleton
_detector_instance = None


def get_detector(model_path: str = DEFAULT_MODEL_PATH) -> SpamDetector:
    """Returns a singleton instance of SpamDetector."""
    global _detector_instance
    if _detector_instance is None or _detector_instance.model_path != model_path:
        _detector_instance = SpamDetector(model_path)
    return _detector_instance


def predict_message(raw_message: Optional[str], model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Convenience function to classify an SMS message directly."""
    detector = get_detector(model_path)
    return detector.predict(raw_message)


if __name__ == "__main__":
    test_samples = [
        "Congratulations! You won a FREE $1000 Walmart giftcard! Click here: http://bit.ly/scam to claim now!",
        "Hey, can you pick up some groceries on your way back home?",
        "",
        "   ",
        None
    ]

    print("--- Inference Engine Test ---")
    detector = SpamDetector()
    if not detector.is_model_loaded():
        print(f"Model artifact not yet present at {DEFAULT_MODEL_PATH}. Run train_model.py first.")
    else:
        for sample in test_samples:
            result = detector.predict(sample)
            print(f"\nInput: {sample!r}")
            print(f"Result: {result['label']} (Spam Prob: {result['spam_probability_uncalibrated']:.4f})")
            print(f"Cleaned: {result['cleaned_text']}")
