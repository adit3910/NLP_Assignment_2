# Statistical Machine Translation (SMT) with BLEU Evaluation

## Project Overview

This project implements a Statistical Machine Translation (SMT) system and evaluates the translation quality using the BLEU (Bilingual Evaluation Understudy) score.

The system:

- Trains a word-based SMT model using a parallel corpus
- Translates input text using learned probabilities
- Evaluates translation quality using manually implemented BLEU metrics
- Provides a Streamlit-based user interface

## Objectives

- Build an SMT system from scratch
- Compute BLEU score without using external BLEU libraries
- Display SMT output, BLEU score, brevity penalty, and 1–4 gram precision table
- Ensure modular and explainable implementation

## Project Structure
```text
SMT_BLEU_Project/
│
├── app/
│ └── app.py
│
├── data/
│ ├── train/
│ │ ├── source.txt
│ │ └── target.txt
│ │
│ └── test/
│ ├── source_test.txt
│ └── reference.txt
│
├── results/
│ ├── translations.txt
│ └── bleu_scores.csv
│
├── src/
│ ├── preprocessing/
│ │ ├── clean_text.py
│ │ └── tokenizer.py
│ │
│ ├── translation/
│ │ ├── translation_model.py
│ │ ├── language_model.py
│ │ └── decoder.py
│ │
│ └── evaluation/
│ ├── ngram_precision.py
│ └── bleu_score.py
│
├── requirements.txt
└── README.md
```

## Dataset Description

Training Data (data/train/):

- source.txt: Source language sentences
- target.txt: Target language translations
- Line-by-line aligned parallel corpus

Test Data (data/test/):

- source_test.txt: Sentences to translate
- reference.txt: Reference translations for BLEU evaluation

## Installation and Setup

1. (Optional) Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

## Running the Application

streamlit run app/app.py

The application opens in a browser and allows input of source text, reference translation, SMT output display, and BLEU score evaluation.

## Methodology

SMT Model:

- Word-based translation model (IBM Model-1 style)
- Learns probabilities P(target | source)

Language Model:

- Bigram language model
- Uses log probabilities for fluency scoring

Decoding:

- Greedy decoding using translation probabilities
- Handles out-of-vocabulary words safely

BLEU Evaluation:

- Modified n-gram precision (1–4 grams)
- Brevity Penalty (BP)
- BLEU = BP × exp((1/N) × Σ log pₙ)

## Output

The system displays:

- Translated sentence
- BLEU score
- Brevity penalty
- N-gram precision table

## Key Features

- No external BLEU libraries used
- Fully modular code
- Easy to explain and reproduce
- Suitable for academic evaluation and viva

## Notes

- A small parallel corpus (5–10 sentence pairs) is sufficient for demonstration
- BLEU score is computed using a single reference translation

## Author

Adithya Chakrala

## Final Status

SMT implemented
BLEU implemented manually
UI completed
Assignment ready
