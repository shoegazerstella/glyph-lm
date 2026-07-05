"""BPE tokenizer training for the raw and glyph corpora (see docs/prd.md)."""

import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from glyph.encoder import get_special_tokens

# GLYPH NOTE: a small vocab_size is intentional — it keeps the model tiny and, more
# importantly, keeps the comparison fair.
VOCAB_SIZE = 2048
BASE_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe(corpus_path: str, save_dir: str, special_tokens: list[str] = BASE_SPECIAL_TOKENS) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
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
    # GLYPH NOTE: registering glyph symbols as trainer special_tokens makes them
    # guaranteed atomic vocab entries — the tokenizer's added-vocabulary matching
    # intercepts them before pre-tokenization/merges run, so they no longer have to
    # *earn* a slot through frequency-based merges, where they previously lost out
    # to common English morphemes (see the worse compression ratio in README.md's
    # first-run results).
    glyph_tok = train_bpe(
        "data/glyph/train.txt",
        "tokenizers/glyph_bpe",
        special_tokens=BASE_SPECIAL_TOKENS + get_special_tokens(),
    )

    report_stats(raw_tok, "data/raw/train.txt", "raw")
    report_stats(glyph_tok, "data/glyph/train.txt", "glyph")
