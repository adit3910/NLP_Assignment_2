"""
translation_model.py
---------------------
This module implements a simple Statistical Machine Translation
model using word-to-word translation probabilities (IBM Model 1 style).
"""

from collections import defaultdict
from typing import List, Dict

from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenizer import tokenize


def load_parallel_corpus(source_file: str, target_file: str):
    """
    Load and preprocess parallel corpus.

    Args:
        source_file (str): Path to source language file
        target_file (str): Path to target language file

    Returns:
        List[List[str]], List[List[str]]: Tokenized source and target sentences
    """
    with open(source_file, "r", encoding="utf-8") as sf, \
         open(target_file, "r", encoding="utf-8") as tf:

        source_lines = sf.readlines()
        target_lines = tf.readlines()

    assert len(source_lines) == len(target_lines), \
        "Source and target files must have same number of lines"

    src_sentences = []
    tgt_sentences = []

    for src, tgt in zip(source_lines, target_lines):
        src_clean = clean_text(src.strip())
        tgt_clean = clean_text(tgt.strip())

        src_sentences.append(tokenize(src_clean))
        tgt_sentences.append(tokenize(tgt_clean))

    return src_sentences, tgt_sentences


def train_translation_model(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Train word-to-word translation probabilities.

    Args:
        src_sentences (List[List[str]]): Tokenized source sentences
        tgt_sentences (List[List[str]]): Tokenized target sentences

    Returns:
        Dict[str, Dict[str, float]]:
        Translation probabilities P(target | source)
    """
    # Count co-occurrences
    co_occurrence = defaultdict(lambda: defaultdict(int))
    source_counts = defaultdict(int)

    for src_tokens, tgt_tokens in zip(src_sentences, tgt_sentences):
        for src_word in src_tokens:
            for tgt_word in tgt_tokens:
                co_occurrence[src_word][tgt_word] += 1
                source_counts[src_word] += 1

    # Normalize counts to probabilities
    translation_probs = defaultdict(dict)
    for src_word in co_occurrence:
        for tgt_word in co_occurrence[src_word]:
            translation_probs[src_word][tgt_word] = (
                co_occurrence[src_word][tgt_word] / source_counts[src_word]
            )

    return translation_probs


# Simple test (run this file directly)
if __name__ == "__main__":
    SOURCE_FILE = "data/train/source.txt"
    TARGET_FILE = "data/train/target.txt"

    src_sents, tgt_sents = load_parallel_corpus(SOURCE_FILE, TARGET_FILE)
    model = train_translation_model(src_sents, tgt_sents)

    # Print sample probabilities
    print("Sample translation probabilities:\n")
    for src_word in list(model.keys())[:5]:
        print(f"{src_word} -> {model[src_word]}")
