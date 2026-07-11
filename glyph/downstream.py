"""Downstream evaluation tasks for GlyphLM.

Next-word accuracy tests whether models learned language patterns vs memorizing
training data. Unlike perplexity (per-token metric), this compares decoded
predictions to ground truth, normalizing across tokenizations.
"""

import random
import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from glyph import encoder


def next_word_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    val_text: str,
    is_glyph: bool,
    n_samples: int = 100,
    context_len: int = 32,
) -> float:
    """Compute next-word prediction accuracy on validation text.

    Args:
        model: Trained language model
        tokenizer: Tokenizer corresponding to model
        val_text: Validation text (raw, not encoded)
        is_glyph: Whether model was trained on glyph-encoded text
        n_samples: Number of random samples to evaluate
        context_len: Number of context words to use for prediction

    Returns:
        Accuracy as float in [0.0, 1.0] (fraction of correct predictions)
    """
    model.eval()

    # Split validation text by whitespace
    words = val_text.split()

    if len(words) <= context_len:
        raise ValueError(f"val_text has {len(words)} words, need at least {context_len + 1}")

    # Sample random indices (range: context_len to len(words))
    valid_indices = list(range(context_len, len(words)))
    sample_indices = random.sample(valid_indices, min(n_samples, len(valid_indices)))

    correct = 0
    total = 0

    for idx in sample_indices:
        # Extract context and ground truth
        context_words = words[idx - context_len:idx]
        ground_truth = words[idx]

        # Build context text
        context_text = " ".join(context_words)

        # For glyph model: encode context before tokenizing
        if is_glyph:
            context_text = encoder.encode(context_text)

        # Tokenize context
        input_ids = tokenizer.encode(context_text, return_tensors="pt")

        # Generate 1 token (greedy)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated token (only the new token)
        generated_ids = output[0, input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # For glyph model: decode prediction from shorthand
        if is_glyph:
            prediction = encoder.decode(prediction)

        # Normalize: case-insensitive, strip punctuation
        prediction_clean = _normalize_word(prediction)
        ground_truth_clean = _normalize_word(ground_truth)

        # Compare
        if prediction_clean == ground_truth_clean:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def _normalize_word(word: str) -> str:
    """Normalize word for comparison: lowercase, strip punctuation."""
    # Strip punctuation: .,!?;:
    word = re.sub(r"[.,!?;:]", "", word)
    # Convert to lowercase
    word = word.lower()
    # Strip whitespace
    word = word.strip()
    return word
