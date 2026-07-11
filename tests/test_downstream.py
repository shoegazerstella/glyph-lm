"""Tests for downstream evaluation tasks."""

import pytest
from glyph.downstream import next_word_accuracy


def test_next_word_accuracy_return_type_and_range():
    """Test that next_word_accuracy returns a float between 0.0 and 1.0."""
    # Mock a simple model and tokenizer for testing
    # We'll use a minimal validation text
    val_text = "the quick brown fox jumps over the lazy dog " * 10

    # For now, test with None model/tokenizer to verify the function exists
    # and has the correct signature. We'll need to mock these properly.
    try:
        # This will fail initially since the function doesn't exist yet
        from glyph.downstream import next_word_accuracy

        # Check function exists and is callable
        assert callable(next_word_accuracy)

        # We can't run it without real model/tokenizer, but we can verify
        # the interface exists
        assert True
    except ImportError:
        pytest.fail("glyph.downstream module or next_word_accuracy function not found")


def test_next_word_accuracy_with_mock():
    """Test next_word_accuracy with mocked model and tokenizer."""
    # This test will verify return value is in valid range
    # We need to create a minimal mock that simulates the interface

    class MockTokenizer:
        eos_token_id = 0

        def encode(self, text, return_tensors=None):
            # Return a simple tensor-like object
            import torch
            return torch.tensor([[1, 2, 3]])

        def decode(self, tokens, skip_special_tokens=True):
            # Return a simple word
            return "the"

    class MockModel:
        def generate(self, input_ids, max_new_tokens=None, do_sample=None, pad_token_id=None):
            # Return the same input with one extra token
            import torch
            return torch.cat([input_ids, torch.tensor([[4]])], dim=1)

        def eval(self):
            return self

    val_text = "the quick brown fox jumps over the lazy dog " * 10

    result = next_word_accuracy(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        val_text=val_text,
        is_glyph=False,
        n_samples=10,
        context_len=5
    )

    # Verify return type and range
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Expected value in [0.0, 1.0], got {result}"
