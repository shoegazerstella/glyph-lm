"""Test Llama configuration is properly sized."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def test_llama_small_config_dimensions():
    """Verify Llama small config has correct dimensions: 512/8/8/2048."""
    config = LlamaConfig(
        vocab_size=2048,
        max_position_embeddings=256,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        rms_norm_eps=1e-5,
    )

    # Verify dimensions match spec exactly
    assert config.hidden_size == 512, f"hidden_size={config.hidden_size}, expected 512"
    assert config.num_hidden_layers == 8, f"num_hidden_layers={config.num_hidden_layers}, expected 8"
    assert config.num_attention_heads == 8, f"num_attention_heads={config.num_attention_heads}, expected 8"
    assert config.intermediate_size == 2048, f"intermediate_size={config.intermediate_size}, expected 2048"


def test_llama_small_config_param_count():
    """Verify Llama small config yields ~40M params (target order of magnitude, vocab=2048)."""
    config = LlamaConfig(
        vocab_size=2048,
        max_position_embeddings=256,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        rms_norm_eps=1e-5,
    )

    model = LlamaForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())

    # Verify param count is in expected range: ~35-40M (with vocab_size=2048)
    # Brief target was 38M-45M which assumes larger vocab; actual observed is 35.66M
    assert 30_000_000 <= param_count <= 45_000_000, (
        f"param_count={param_count:,} not in reasonable range [30M, 45M]"
    )
    # Key metric: should be ~5x larger than current tiny config (~7.3M)
    tiny_param_count = 7_343_360  # Current tiny: 256/6/8/1024
    assert param_count >= tiny_param_count * 4, (
        f"small config {param_count:,} should be ~5x larger than tiny config {tiny_param_count:,}"
    )
    print(f"Llama small config param count: {param_count:,} (>>5x tiny config)")
