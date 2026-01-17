"""
app.py
-------
Streamlit UI for Statistical Machine Translation (SMT)
with BLEU score evaluation.
"""

import streamlit as st
import sys
from pathlib import Path

# Get project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Add project root to Python path
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenizer import tokenize
from src.translation.translation_model import (
    load_parallel_corpus,
    train_translation_model,
)
from src.translation.language_model import train_bigram_language_model
from src.translation.decoder import decode_sentence
from src.evaluation.bleu_score import compute_bleu_score


# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SMT with BLEU Evaluation",
    layout="centered"
)

st.title("📘 Statistical Machine Translation (SMT)")
st.subheader("with BLEU Score Evaluation")

st.markdown(
    """
This application translates text using a simple SMT system
and evaluates the translation quality using BLEU score.
"""
)

# --------------------------------------------------
# Load & Train Models (Once)
# --------------------------------------------------
@st.cache_data
def load_models():
    source_file = "data/train/source.txt"
    target_file = "data/train/target.txt"

    src_sentences, tgt_sentences = load_parallel_corpus(
        source_file, target_file
    )

    translation_model = train_translation_model(
        src_sentences, tgt_sentences
    )

    language_model = train_bigram_language_model(tgt_sentences)

    # -------- LOGGING --------
    print("\n=== SMT TRAINING LOG ===")
    print(f"Training sentence pairs: {len(src_sentences)}")

    src_vocab = set(word for sent in src_sentences for word in sent)
    tgt_vocab = set(word for sent in tgt_sentences for word in sent)

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    # Show sample learned translations
    sample_words = list(translation_model.keys())[:5]
    print("Sample translation probabilities:")
    for word in sample_words:
        print(f"  {word} -> {translation_model[word]}")

    print("=== TRAINING COMPLETE ===\n")
    # -------------------------

    return translation_model, language_model


translation_model, language_model = load_models()

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.header("🔤 Input")

source_text = st.text_area(
    "Enter source sentence:",
    height=100
)

reference_text = st.text_area(
    "Enter reference translation (for BLEU):",
    height=100
)

# --------------------------------------------------
# Translate Button
# --------------------------------------------------
if st.button("🚀 Translate"):
    if not source_text.strip():
        st.warning("Please enter source text.")
    else:
        # Preprocess source
        clean_src = clean_text(source_text)
        src_tokens = tokenize(clean_src)

        # Decode translation
        translated_tokens = decode_sentence(
            src_tokens, translation_model, language_model
        )
        translated_text = " ".join(translated_tokens)

        st.header("📝 SMT Output")
        st.success(translated_text)

        # BLEU Evaluation
        if reference_text.strip():
            clean_ref = clean_text(reference_text)
            ref_tokens = tokenize(clean_ref)

            bleu_results = compute_bleu_score(
                translated_tokens, ref_tokens
            )

            st.header("📊 BLEU Evaluation")

            st.metric(
                label="BLEU Score",
                value=f"{bleu_results['bleu']:.4f}"
            )

            st.write("**Brevity Penalty:**",
                     f"{bleu_results['brevity_penalty']:.4f}")

            st.subheader("N-gram Precision Table")
            st.table({
                "N-gram": ["1-gram", "2-gram", "3-gram", "4-gram"],
                "Precision": [
                    bleu_results["1-gram"],
                    bleu_results["2-gram"],
                    bleu_results["3-gram"],
                    bleu_results["4-gram"],
                ],
            })
        else:
            st.info("Reference translation not provided. BLEU not computed.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "📌 **NLP Assignment – Statistical Machine Translation with BLEU Evaluation**"
)
