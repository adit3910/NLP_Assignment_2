"""
decoder.py
-----------
This module decodes a source sentence into a target sentence
using translation probabilities and a language model.
"""

from typing import List, Dict, Tuple

from src.translation.language_model import score_sentence


def decode_sentence(
    src_tokens: List[str],
    translation_probs: Dict[str, Dict[str, float]],
    language_model: Dict[Tuple[str, str], float]
) -> List[str]:
    """
    Decode a source sentence using greedy decoding.

    Args:
        src_tokens (List[str]): Tokenized source sentence
        translation_probs (Dict): P(target | source)
        language_model (Dict): Bigram language model

    Returns:
        List[str]: Decoded target sentence tokens
    """
    target_tokens = []

    for src_word in src_tokens:
        if src_word in translation_probs:
            # Pick target word with highest translation probability
            best_target = max(
                translation_probs[src_word],
                key=translation_probs[src_word].get
            )
            target_tokens.append(best_target)
        else:
            # OOV word fallback
            target_tokens.append(src_word)

    return target_tokens


def decode_with_lm(
    src_tokens: List[str],
    translation_probs: Dict[str, Dict[str, float]],
    language_model: Dict[Tuple[str, str], float]
) -> List[str]:
    """
    Decode using translation model + language model scoring.

    (Simple extension for fluency)

    Args:
        src_tokens (List[str]): Source tokens
        translation_probs (Dict): Translation probabilities
        language_model (Dict): Bigram LM

    Returns:
        List[str]: Decoded target sentence
    """
    # Initial greedy decoding
    candidate = decode_sentence(src_tokens, translation_probs, language_model)

    # Score with language model (can be extended)
    _ = score_sentence(candidate, language_model)

    return candidate


# Simple test (run this file directly)
if __name__ == "__main__":
    src_tokens = ["hello", "world"]

    translation_probs = {
        "hello": {"namaste": 0.8, "hello": 0.2},
        "world": {"duniya": 1.0}
    }

    language_model = {
        ("namaste", "duniya"): -0.1
    }

    decoded = decode_with_lm(src_tokens, translation_probs, language_model)
    print("Source Tokens :", src_tokens)
    print("Decoded Tokens:", decoded)
