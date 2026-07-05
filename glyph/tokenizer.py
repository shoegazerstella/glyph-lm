"""BPE tokenizer training for the raw and glyph corpora (see docs/prd.md)."""

import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# GLYPH NOTE: a small vocab_size is intentional — it keeps the model tiny and, more
# importantly, keeps the comparison fair. The glyph corpus introduces novel symbols
# (þ, ŋ, ʃ, ...) that must earn their place in a fixed-size vocab alongside ordinary
# subwords; how well BPE allocates that budget is itself part of what's being tested.
VOCAB_SIZE = 2048
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe(corpus_path: str, save_dir: str) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tokenizer.train([corpus_path], trainer)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    return tokenizer


def report_stats(tokenizer: Tokenizer, corpus_path: str, label: str) -> None:
    with open(corpus_path) as f:
        lines = [line for line in f.read().split("\n") if line.strip()]

    encodings = tokenizer.encode_batch(lines)
    total_tokens = sum(len(e.ids) for e in encodings)
    total_chars = sum(len(line) for line in lines)

    print(f"[{label}] vocab_size={tokenizer.get_vocab_size()}")
    print(f"[{label}] avg tokens/line={total_tokens / len(lines):.2f}")
    print(f"[{label}] compression ratio vs chars={total_tokens / total_chars:.4f}")


if __name__ == "__main__":
    raw_tok = train_bpe("data/raw/train.txt", "tokenizers/raw_bpe")
    glyph_tok = train_bpe("data/glyph/train.txt", "tokenizers/glyph_bpe")

    report_stats(raw_tok, "data/raw/train.txt", "raw")
    report_stats(glyph_tok, "data/glyph/train.txt", "glyph")
