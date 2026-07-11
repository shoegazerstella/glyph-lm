"""BPE tokenizer training for the raw and glyph corpora (see docs/prd.md)."""

import os
import tempfile

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from glyph.encoder import get_special_tokens

# GLYPH NOTE: increased from 2048 to 4096 to reduce BPE vocab starvation on larger corpus
VOCAB_SIZE = 4096
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


def train_sentencepiece(corpus_path: str, save_dir: str, special_tokens: list[str] = BASE_SPECIAL_TOKENS) -> Tokenizer:
    """Train SentencePiece Unigram tokenizer and convert to tokenizers.Tokenizer format.

    Args:
        corpus_path: Path to training corpus text file
        save_dir: Directory to save tokenizer.json
        special_tokens: List of special tokens (BASE_SPECIAL_TOKENS + shorthand chords)

    Returns:
        Tokenizer instance loaded from saved tokenizer.json
    """
    os.makedirs(save_dir, exist_ok=True)

    # SentencePiece requires model_prefix for output files
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "spm")

        # Filter out control symbols that SentencePiece defines internally
        # (unk, pad, bos, eos are set via *_id params, not user_defined_symbols)
        control_symbols = {"<unk>", "<pad>", "<bos>", "<eos>"}
        user_symbols = [tok for tok in special_tokens if tok not in control_symbols]

        # Train SentencePiece unigram model
        # user_defined_symbols makes shorthand chords atomic (not split during training)
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=VOCAB_SIZE,
            character_coverage=1.0,
            unk_id=3,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            user_defined_symbols=user_symbols,
        )

        # Load trained SentencePiece model
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")

        # Convert to tokenizers.Tokenizer format via JSON export/import
        # SentencePiece .model format is not directly compatible with transformers,
        # so we build a tokenizers.Tokenizer with the same vocabulary
        from tokenizers import Tokenizer as TokenizerBase
        from tokenizers.models import Unigram

        # Extract vocabulary from SentencePiece model
        vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]

        # Create Unigram tokenizer with extracted vocabulary
        tokenizer = TokenizerBase(Unigram(vocab, unk_id=sp.unk_id()))

        # Save to tokenizer.json
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
    import argparse

    parser = argparse.ArgumentParser(description="Train tokenizers for raw and glyph corpora")
    parser.add_argument(
        "--type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Tokenizer type to train (default: bpe)",
    )
    args = parser.parse_args()

    train_fn = train_bpe if args.type == "bpe" else train_sentencepiece
    tokenizer_type = "bpe" if args.type == "bpe" else "unigram"

    raw_tok = train_fn("data/raw/train.txt", f"tokenizers/raw_{tokenizer_type}")
    # GLYPH NOTE: registering glyph symbols as trainer special_tokens makes them
    # guaranteed atomic vocab entries — the tokenizer's added-vocabulary matching
    # intercepts them before pre-tokenization/merges run, so they no longer have to
    # *earn* a slot through frequency-based merges, where they previously lost out
    # to common English morphemes (see the worse compression ratio in README.md's
    # first-run results).
    glyph_tok = train_fn(
        "data/glyph/train.txt",
        f"tokenizers/glyph_{tokenizer_type}",
        special_tokens=BASE_SPECIAL_TOKENS + get_special_tokens(),
    )

    report_stats(raw_tok, "data/raw/train.txt", "raw")
    report_stats(glyph_tok, "data/glyph/train.txt", "glyph")
