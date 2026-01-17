"""
language_model.py
------------------
This module implements an n-gram Language Model (Bigram)
to estimate sentence fluency for SMT decoding.
"""

from collections import defaultdict
from typing import List, Dict, Tuple
import math


def train_bigram_language_model(
    sentences: List[List[str]]
) -> Dict[Tuple[str, str], float]:
    """
    Train a bigram language model using log-probabilities.

    Args:
        sentences (List[List[str]]): Tokenized sentences (target language)

    Returns:
        Dict[Tuple[str, str], float]: Bigram log-probabilities
    """
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)

    # Count unigrams and bigrams
    for tokens in sentences:
        if not tokens:
            continue

        for i in range(len(tokens)):
            unigram_counts[tokens[i]] += 1

            if i < len(tokens) - 1:
                bigram_counts[(tokens[i], tokens[i + 1])] += 1

    # Compute log-probabilities
    bigram_model = {}
    for (w1, w2), count in bigram_counts.items():
        probability = count / unigram_counts[w1]
        bigram_model[(w1, w2)] = math.log(probability)

    return bigram_model


def score_sentence(
    tokens: List[str],
    bigram_model: Dict[Tuple[str, str], float],
    default_log_prob: float = -10.0
) -> float:
    """
    Score a sentence using the bigram language model.

    Args:
        tokens (List[str]): Tokenized sentence
        bigram_model (Dict): Trained bigram LM
        default_log_prob (float): Penalty for unseen bigrams

    Returns:
        float: Log-probability score
    """
    score = 0.0

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        score += bigram_model.get(bigram, default_log_prob)

    return score


# Simple test (run this file directly)
if __name__ == "__main__":
    sample_sentences = [
        ["this", "is", "a", "test"],
        ["this", "is", "another", "test"],
        ["language", "model", "test"]
    ]

    lm = train_bigram_language_model(sample_sentences)

    test_sentence = ["this", "is", "a", "test"]
    score = score_sentence(test_sentence, lm)

    print("Bigram Language Model:")
    for k, v in list(lm.items())[:5]:
        print(f"{k}: {v}")

    print("\nSentence Score:", score)
