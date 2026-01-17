"""
ngram_precision.py
------------------
This module computes modified n-gram precision
used in BLEU score calculation.
"""

from collections import Counter
from typing import List, Tuple


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Extract n-grams from a token list.

    Args:
        tokens (List[str]): Tokenized sentence
        n (int): n-gram size

    Returns:
        Counter: n-gram counts
    """
    ngrams = Counter()

    if len(tokens) < n or n <= 0:
        return ngrams

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1

    return ngrams


def modified_ngram_precision(
    candidate_tokens: List[str],
    reference_tokens: List[str],
    n: int
) -> float:
    """
    Compute modified n-gram precision.

    Args:
        candidate_tokens (List[str]): SMT output tokens
        reference_tokens (List[str]): Reference translation tokens
        n (int): n-gram size

    Returns:
        float: Modified n-gram precision
    """
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    reference_ngrams = get_ngrams(reference_tokens, n)

    if not candidate_ngrams:
        return 0.0

    clipped_count = 0
    for ngram in candidate_ngrams:
        clipped_count += min(
            candidate_ngrams[ngram],
            reference_ngrams.get(ngram, 0)
        )

    total_count = sum(candidate_ngrams.values())

    return clipped_count / total_count


# Simple test (run this file directly)
if __name__ == "__main__":
    candidate = ["the", "cat", "is", "on", "the", "mat"]
    reference = ["the", "cat", "sat", "on", "the", "mat"]

    for n in range(1, 5):
        precision = modified_ngram_precision(candidate, reference, n)
        print(f"{n}-gram precision: {precision:.4f}")
