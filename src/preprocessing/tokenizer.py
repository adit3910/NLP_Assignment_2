"""
tokenizer.py
----------------
This module tokenizes cleaned text into word-level tokens.
It is used by translation and BLEU evaluation modules.
"""

from typing import List


def tokenize(sentence: str) -> List[str]:
    """
    Tokenize a cleaned sentence into words.

    Args:
        sentence (str): Cleaned input sentence

    Returns:
        List[str]: List of word tokens
    """
    if not sentence:
        return []

    return sentence.split()


def tokenize_sentences(sentences: List[str]) -> List[List[str]]:
    """
    Tokenize multiple sentences.

    Args:
        sentences (List[str]): List of cleaned sentences

    Returns:
        List[List[str]]: Tokenized sentences
    """
    return [tokenize(sentence) for sentence in sentences]


# Simple test (run this file directly)
if __name__ == "__main__":
    sample_sentence = "hello world nlp assignment"
    sample_sentences = [
        "machine translation is fun",
        "bleu score evaluation"
    ]

    print("Single sentence tokens:")
    print(tokenize(sample_sentence))

    print("\nMultiple sentences tokens:")
    print(tokenize_sentences(sample_sentences))
