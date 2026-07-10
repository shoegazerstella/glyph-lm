"""Corpus download + shorthand encoding (see docs/prd.md)."""

import os
import sys

import requests

from glyph import encoder

RAW_DIR = "data/raw"
GLYPH_DIR = "data/glyph"

# Project Gutenberg Gothic Fiction bookshelf (https://www.gutenberg.org/ebooks/bookshelf/638)
GOTHIC_FICTION_IDS = [
    345, 84, 8492, 72, 19476, 22541, 45839, 40284, 18857, 2825,
    26563, 1514, 55, 68283, 17157, 70652, 14287, 10002, 4791,
    68957, 50133, 69608, 50290, 18151, 30434
]


def load_corpus_gothic() -> tuple[str, str]:
    """Download Gothic Fiction books from Project Gutenberg (~8MB actual size)."""
    target_mb = 100
    all_texts = []
    total_mb = 0

    for book_id in GOTHIC_FICTION_IDS:
        if total_mb >= target_mb:
            break

        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        print(f"Downloading book {book_id} ({url})...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text

            # Strip Gutenberg header/footer
            start_marker = "*** START OF"
            end_marker = "*** END OF"
            if start_marker in text:
                parts = text.split(start_marker, 1)
                if len(parts) > 1:
                    text = parts[1].split('\n', 1)[1] if '\n' in parts[1] else parts[1]
            if end_marker in text:
                text = text.split(end_marker, 1)[0]

            text = text.strip()
            if text:
                all_texts.append(text)
                size_mb = len(text) / (1024 * 1024)
                total_mb += size_mb
                print(f"  ✓ Added {size_mb:.2f} MB (total: {total_mb:.2f} MB)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    if not all_texts:
        raise RuntimeError("Failed to download any books from Gutenberg")

    full_text = "\n\n".join(all_texts)
    print(f"\nCorpus ready: {len(full_text) / (1024*1024):.2f} MB, {len(full_text):,} chars")

    # 90/10 train/val split
    split_idx = int(len(full_text) * 0.9)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    return train_text, val_text


def load_corpus_shakespeare() -> tuple[str, str]:
    """Download tiny_shakespeare directly from Andrej's char-rnn repo (~1MB)."""
    print("Downloading tiny_shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    full_text = response.text

    print(f"Corpus ready: {len(full_text) / (1024*1024):.2f} MB, {len(full_text):,} chars")

    # 90/10 train/val split
    split_idx = int(len(full_text) * 0.9)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    return train_text, val_text


def load_corpus(corpus: str = "gothic") -> tuple[str, str]:
    """Load corpus by name: 'gothic' or 'shakespeare'."""
    if corpus == "shakespeare":
        return load_corpus_shakespeare()
    elif corpus == "gothic":
        return load_corpus_gothic()
    else:
        raise ValueError(f"Unknown corpus: {corpus}. Use 'gothic' or 'shakespeare'.")


def build_corpora(corpus: str = "gothic") -> None:
    """Build raw and glyph corpora from selected corpus."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(GLYPH_DIR, exist_ok=True)

    train_text, val_text = load_corpus(corpus)

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
    corpus = sys.argv[1] if len(sys.argv) > 1 else "gothic"
    print(f"Building corpus: {corpus}")
    build_corpora(corpus)
