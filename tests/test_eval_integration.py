"""Integration tests for eval.py output structure."""

import json
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest
import torch
from transformers import PreTrainedTokenizerFast


def test_results_json_includes_next_word_accuracy():
    """Verify that results.json includes next_word_accuracy key for both raw and glyph models."""

    # Mock model and tokenizer
    class MockModel(torch.nn.Module):
        def eval(self):
            return self

        def __call__(self, input_ids=None, labels=None):
            out = MagicMock()
            out.loss = torch.tensor(0.5)
            return out

        def generate(self, input_ids, max_new_tokens=None, do_sample=None, pad_token_id=None, top_k=None, temperature=None):
            # Return input plus some generated tokens
            batch_size = input_ids.shape[0]
            gen_tokens = torch.randint(0, 100, (batch_size, max_new_tokens if max_new_tokens else 10))
            return torch.cat([input_ids, gen_tokens], dim=1)

    class MockTokenizer:
        eos_token_id = 0

        def encode(self, text, return_tensors=None):
            # Return enough tokens to avoid division by zero
            # Create a list of 1000 tokens so compute_perplexity has enough
            tokens = list(range(1, 1001))
            if return_tensors == "pt":
                return torch.tensor([tokens])
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            return "test word"

    # Mock validation text
    val_text = "the quick brown fox jumps over the lazy dog " * 10

    # Mock the main function dependencies
    with patch('glyph.eval.get_device') as mock_get_device, \
         patch('glyph.eval.PreTrainedTokenizerFast') as mock_tokenizer_cls, \
         patch('glyph.eval.AutoModelForCausalLM.from_pretrained') as mock_model_loader, \
         patch('glyph.eval.next_word_accuracy') as mock_nwa, \
         patch('glyph.eval.encode', return_value='encoded_text') as mock_encode, \
         patch('glyph.eval.decode', return_value='decoded_text') as mock_decode, \
         patch('builtins.open', create=True) as mock_open, \
         patch('json.dump') as mock_json_dump, \
         patch('builtins.print'):

        mock_get_device.return_value = 'cpu'
        mock_tokenizer_cls.return_value = MockTokenizer()
        mock_model_loader.return_value = MockModel()
        mock_nwa.return_value = 0.75  # dummy accuracy value

        # Mock file reads
        file_contents = {
            'data/raw/val.txt': val_text,
            'data/glyph/val.txt': val_text,
            'models/model_raw/train_metrics.json': '{"final_loss": 1.0, "train_time_seconds": 100}',
            'models/model_glyph/train_metrics.json': '{"final_loss": 0.9, "train_time_seconds": 105}',
        }

        def mock_open_impl(path, mode='r'):
            from unittest.mock import mock_open as builtin_mock_open
            if path in file_contents:
                return builtin_mock_open(read_data=file_contents[path])()
            return builtin_mock_open()()

        mock_open.side_effect = mock_open_impl

        # Capture the results dict passed to json.dump
        captured_results = {}
        def capture_json_dump(obj, f, **kwargs):
            captured_results['results'] = obj

        mock_json_dump.side_effect = capture_json_dump

        # Import and run main
        from glyph.eval import main
        main()

        # Verify results.json structure
        results = captured_results.get('results', {})

        # Check that both models are present
        assert 'raw' in results, "results.json should contain 'raw' model"
        assert 'glyph' in results, "results.json should contain 'glyph' model"

        # Check that next_word_accuracy is present for both
        assert 'next_word_accuracy' in results['raw'], \
            "results['raw'] should contain 'next_word_accuracy' key"
        assert 'next_word_accuracy' in results['glyph'], \
            "results['glyph'] should contain 'next_word_accuracy' key"

        # Check that values are floats in [0, 1]
        assert isinstance(results['raw']['next_word_accuracy'], float), \
            "next_word_accuracy should be a float"
        assert isinstance(results['glyph']['next_word_accuracy'], float), \
            "next_word_accuracy should be a float"
        assert 0.0 <= results['raw']['next_word_accuracy'] <= 1.0, \
            "next_word_accuracy should be in [0.0, 1.0]"
        assert 0.0 <= results['glyph']['next_word_accuracy'] <= 1.0, \
            "next_word_accuracy should be in [0.0, 1.0]"
