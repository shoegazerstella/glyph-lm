"""Corpus download + shorthand encoding (see docs/prd.md)."""

import os

from datasets import load_dataset

from glyph.encoder import encode

RAW_DIR = "data/raw"
GLYPH_DIR = "data/glyph"


def load_corpus() -> tuple[str, str]:
    # GLYPH NOTE: karpathy/tiny_shakespeare already ships train/validation/test
    # splits at a roughly 90/10 train/val ratio. We use `train` as the single corpus
    # for both tokenizer training and model training, and hold `validation` out
    # strictly for eval.py's perplexity computation. This satisfies "don't reuse the
    # same split for training and eval" without a manual re-split that would just
    # shrink the training corpus for no added safety.
    ds = load_dataset("karpathy/tiny_shakespeare")
    train_text = ds["train"]["text"][0]
    val_text = ds["validation"]["text"][0]
    return train_text, val_text


def build_corpora() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(GLYPH_DIR, exist_ok=True)

    train_text, val_text = load_corpus()

    lines = [line for line in train_text.split("\n") if line.strip()]
    glyph_lines = [encode(line) for line in lines]

    with open(f"{RAW_DIR}/train.txt", "w") as f:
        f.write("\n".join(lines))
    with open(f"{GLYPH_DIR}/train.txt", "w") as f:
        f.write("\n".join(glyph_lines))

    # Held out for eval.py perplexity only — never used for tokenizer/model training.
    with open(f"{RAW_DIR}/val.txt", "w") as f:
        f.write(val_text)
    with open(f"{GLYPH_DIR}/val.txt", "w") as f:
        f.write(encode(val_text))

    raw_tokens = sum(len(line.split()) for line in lines)
    glyph_tokens = sum(len(line.split()) for line in glyph_lines)
    ratio = glyph_tokens / raw_tokens
    print(f"Whitespace-token compression ratio (glyph/raw): {ratio:.4f}")


if __name__ == "__main__":
    build_corpora()
