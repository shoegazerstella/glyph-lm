"""Corpus download + shorthand encoding (see docs/prd.md)."""

import os

from datasets import load_dataset

from glyph import encoder

RAW_DIR = "data/raw"
GLYPH_DIR = "data/glyph"


def load_corpus() -> tuple[str, str]:
    # GLYPH NOTE: karpathy/tiny_shakespeare already ships train/validation/test
    # splits at a roughly 90/10 train/val ratio. We use `train` as the single corpus
    # for both tokenizer training and model training, and hold `validation` out
    # strictly for eval.py's perplexity computation. This satisfies "don't reuse the
    # same split for training and eval" without a manual re-split that would just
    # shrink the training corpus for no added safety.
    # GLYPH NOTE: the dataset's original loader is a legacy HF "dataset script",
    # which recent `datasets` versions refuse to execute. Pinning to the
    # auto-converted parquet revision gets the identical train/validation/test
    # splits without needing trust_remote_code.
    ds = load_dataset("karpathy/tiny_shakespeare", revision="refs/convert/parquet")
    train_text = ds["train"]["text"][0]
    val_text = ds["validation"]["text"][0]
    return train_text, val_text


def build_corpora() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(GLYPH_DIR, exist_ok=True)

    train_text, val_text = load_corpus()

    # GLYPH NOTE: the shorthand mapping is fitted from the train split only (never
    # from val), then cached to disk — eval.py and chat.py load the same fitted
    # mapping rather than each re-deriving their own from a different corpus slice.
    encoder.fit(train_text)

    lines = [line for line in train_text.split("\n") if line.strip()]
    glyph_lines = [encoder.encode(line) for line in lines]

    with open(f"{RAW_DIR}/train.txt", "w") as f:
        f.write("\n".join(lines))
    with open(f"{GLYPH_DIR}/train.txt", "w") as f:
        f.write("\n".join(glyph_lines))

    # Held out for eval.py perplexity only — never used for tokenizer/model training.
    with open(f"{RAW_DIR}/val.txt", "w") as f:
        f.write(val_text)
    with open(f"{GLYPH_DIR}/val.txt", "w") as f:
        f.write(encoder.encode(val_text))

    raw_tokens = sum(len(line.split()) for line in lines)
    glyph_tokens = sum(len(line.split()) for line in glyph_lines)
    ratio = glyph_tokens / raw_tokens
    print(f"Whitespace-token compression ratio (glyph/raw): {ratio:.4f}")


if __name__ == "__main__":
    build_corpora()
