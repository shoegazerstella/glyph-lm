"""Test unigram tokenizer training."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from glyph.tokenizer import train_sentencepiece, BASE_SPECIAL_TOKENS


def test_train_sentencepiece_creates_tokenizer_json():
    """Verify train_sentencepiece() creates tokenizer.json with unigram model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test corpus with enough diversity for small vocab
        # Repeat content to provide sufficient training data
        corpus_text = "\n".join([
            "hello world this is a test",
            "the quick brown fox jumps over the lazy dog",
            "testing tokenizer training with unigram model",
            "sentencepiece requires sufficient vocabulary coverage",
        ] * 10)  # Repeat 10x for sufficient training data
        corpus_path = Path(tmpdir) / "test_corpus.txt"
        corpus_path.write_text(corpus_text)

        save_dir = Path(tmpdir) / "tokenizer_out"

        # Train unigram tokenizer with smaller vocab for test
        # Mock VOCAB_SIZE to use 50 instead of 4096 (test corpus too small)
        with patch("glyph.tokenizer.VOCAB_SIZE", 50):
            tokenizer = train_sentencepiece(
                str(corpus_path),
                str(save_dir),
                special_tokens=BASE_SPECIAL_TOKENS
            )

        # Verify tokenizer.json was created
        tokenizer_json = save_dir / "tokenizer.json"
        assert tokenizer_json.exists(), f"tokenizer.json not created at {tokenizer_json}"

        # Verify tokenizer is functional
        assert tokenizer is not None
        assert tokenizer.get_vocab_size() > 0

        # Verify control tokens are in vocab (SentencePiece uses <s>, </s>, not <bos>, <eos>)
        vocab = tokenizer.get_vocab()
        sp_control_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
        for token in sp_control_tokens:
            assert token in vocab, f"Control token {token} not in vocab"
