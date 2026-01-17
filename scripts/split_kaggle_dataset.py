import os

# Input file (raw Kaggle/Tatoeba format)
INPUT_FILE = "data/raw/hindi.txt"

# Output files
SOURCE_OUT = "data/train/source.txt"
TARGET_OUT = "data/train/target.txt"

# Ensure output directory exists
os.makedirs("data/train", exist_ok=True)

source_sentences = []
target_sentences = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Split by tab
        parts = line.split("\t")

        # We expect at least: English \t Hindi \t metadata
        if len(parts) < 2:
            continue

        english = parts[0].strip()
        hindi = parts[1].strip()

        # Skip empty entries
        if english and hindi:
            source_sentences.append(english)
            target_sentences.append(hindi)

# Write to training files
with open(SOURCE_OUT, "w", encoding="utf-8") as sf, \
     open(TARGET_OUT, "w", encoding="utf-8") as tf:

    for src, tgt in zip(source_sentences, target_sentences):
        sf.write(src + "\n")
        tf.write(tgt + "\n")

print("Dataset split complete!")
print(f"Total sentence pairs: {len(source_sentences)}")
print(f"Saved to: {SOURCE_OUT}, {TARGET_OUT}")