"""
app.py
-------
Streamlit UI for Statistical Machine Translation (SMT)
with BLEU score evaluation.
"""

import streamlit as st
import sys
from pathlib import Path

# --------------------------------------------------
# Project Path Setup
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
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
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SMT Translator",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }

    .main {
        padding: 2rem;
    }

    h1 {
        font-weight: 800;
        color: #1f2937;
    }

    h4 {
        color: #6b7280;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
    }

    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🌍 Statistical Machine Translation</h1>
    <h4 style='text-align:center;'>
    Translation with BLEU Score Evaluation
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown(
        """
        **Features**
        - IBM Model-1 inspired SMT
        - Bigram Language Model
        - Word-level decoding
        - BLEU score evaluation

        **Evaluation**
        - Brevity Penalty
        - 1-gram to 4-gram precision

        **Course**
        NLP – Assignment 2
        """
    )
    st.divider()
    st.caption("Built using Python & Streamlit")

# --------------------------------------------------
# Load & Train Models (Cached)
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

    return translation_model, language_model


translation_model, language_model = load_models()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.markdown("## 🔤 Input Sentences")

col1, col2 = st.columns(2)

with col1:
    source_text = st.text_area(
        "Source Sentence",
        height=150,
        placeholder="Enter source language sentence here..."
    )

with col2:
    reference_text = st.text_area(
        "Reference Translation (Optional)",
        height=150,
        placeholder="Enter reference translation for BLEU score..."
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Translate Button
# --------------------------------------------------
translate_btn = st.button("🚀 Translate & Evaluate", use_container_width=True)

# --------------------------------------------------
# Translation & Evaluation
# --------------------------------------------------
if translate_btn:
    if not source_text.strip():
        st.warning("⚠️ Please enter a source sentence.")
    else:
        with st.spinner("Translating and evaluating..."):
            # Preprocess
            clean_src = clean_text(source_text)
            src_tokens = tokenize(clean_src)

            # Decode
            translated_tokens = decode_sentence(
                src_tokens, translation_model, language_model
            )
            translated_text = " ".join(translated_tokens)

        # Output
        st.markdown("## 📝 SMT Output")
        st.success(translated_text)

        # BLEU Evaluation
        if reference_text.strip():
            clean_ref = clean_text(reference_text)
            ref_tokens = tokenize(clean_ref)

            bleu_results = compute_bleu_score(
                translated_tokens, ref_tokens
            )

            st.markdown("## 📊 BLEU Evaluation")

            c1, c2, c3 = st.columns(3)

            c1.metric("BLEU Score", f"{bleu_results['bleu']:.4f}")
            c2.metric("Brevity Penalty", f"{bleu_results['brevity_penalty']:.4f}")
            c3.metric("Output Length", len(translated_tokens))

            st.markdown("### 📈 N-gram Precision")

            st.dataframe(
                {
                    "N-gram": ["1-gram", "2-gram", "3-gram", "4-gram"],
                    "Precision": [
                        round(bleu_results["1-gram"], 4),
                        round(bleu_results["2-gram"], 4),
                        round(bleu_results["3-gram"], 4),
                        round(bleu_results["4-gram"], 4),
                    ],
                },
                use_container_width=True
            )
        else:
            st.info("ℹ️ Reference translation not provided. BLEU score not computed.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("📌 NLP Assignment – Statistical Machine Translation with BLEU Evaluation")
