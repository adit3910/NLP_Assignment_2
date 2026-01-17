"""
bleu_score.py
--------------
This module computes the BLEU score for a single candidate
sentence against a single reference sentence.
"""

import math
from typing import List, Dict

from src.evaluation.ngram_precision import modified_ngram_precision


def brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """
    Compute BLEU brevity penalty (BP).

    Args:
        candidate_len (int): Length of candidate sentence
        reference_len (int): Length of reference sentence

    Returns:
        float: Brevity penalty
    """
    if candidate_len == 0:
        return 0.0

    if candidate_len > reference_len:
        return 1.0

    return math.exp(1 - (reference_len / candidate_len))


def compute_bleu_score(
    candidate_tokens: List[str],
    reference_tokens: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score and intermediate values.

    Args:
        candidate_tokens (List[str]): SMT output tokens
        reference_tokens (List[str]): Reference translation tokens
        max_n (int): Maximum n-gram order (default=4)

    Returns:
        Dict[str, float]: BLEU score details
    """
    precisions = []
    for n in range(1, max_n + 1):
        p_n = modified_ngram_precision(candidate_tokens, reference_tokens, n)
        precisions.append(p_n)

    # If any precision is zero, BLEU becomes zero
    if min(precisions) == 0:
        bleu = 0.0
    else:
        log_precision_sum = sum(math.log(p) for p in precisions) / max_n
        bp = brevity_penalty(len(candidate_tokens), len(reference_tokens))
        bleu = bp * math.exp(log_precision_sum)

    return {
        "bleu": bleu,
        "brevity_penalty": brevity_penalty(
            len(candidate_tokens), len(reference_tokens)
        ),
        "1-gram": precisions[0],
        "2-gram": precisions[1],
        "3-gram": precisions[2],
        "4-gram": precisions[3],
    }


# Simple test (run this file directly)
if __name__ == "__main__":
    candidate = ["the", "cat", "is", "on", "the", "mat"]
    reference = ["the", "cat", "sat", "on", "the", "mat"]

    results = compute_bleu_score(candidate, reference)

    print("BLEU Evaluation Results\n")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
