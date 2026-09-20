"""Deterministic text cleaning shared by training and inference.

Tokenization is deliberately left to scikit-learn's TF-IDF vectorizer. We do
not remove stop words because negations (for example, "not" and "never") can
materially alter a political statement's meaning.
"""

import re
from typing import Any

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s']+")
_UNDERSCORE_RE = re.compile(r"_+")
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(value: Any) -> str:
    """Return a normalized, lowercase representation of an input value.

    Steps: coerce valid values to text, remove URLs, lowercase, replace
    punctuation/formatting characters with spaces, and collapse whitespace.
    Apostrophes are retained so contractions and negation remain recognizable.
    """
    if value is None:
        return ""
    try:
        if value != value:  # NaN is the only common value unequal to itself.
            return ""
    except (TypeError, ValueError):
        return ""

    text = str(value).strip()
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _UNDERSCORE_RE.sub(" ", text)
    text = _NON_TEXT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()

