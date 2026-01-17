"""
clean_text.py
--------------
Unicode-safe text normalization for SMT.
"""

import re
import unicodedata


def clean_text(sentence: str) -> str:
    if not sentence:
        return ""

    # Normalize Unicode (VERY IMPORTANT for Hindi)
    sentence = unicodedata.normalize("NFC", sentence)

    # Lowercase (safe for English, no effect on Hindi)
    sentence = sentence.lower()

    # Remove punctuation ONLY
    sentence = re.sub(r"[.,!?;:\"()]", "", sentence)

    # Normalize whitespace
    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence
