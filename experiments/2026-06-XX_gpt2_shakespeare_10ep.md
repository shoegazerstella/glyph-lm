# GPT-2 Shakespeare 10 Epochs (seed=42)

**Date:** ~2026-06 (from README §5.1)  
**Architecture:** GPT-2 (n_embd=128, n_layer=4, n_head=4, n_positions=256)  
**Corpus:** tiny_shakespeare (~1MB)  
**Training:** 10 epochs, AdamW lr=3e-4  
**Seed:** 42  
**Hardware:** Apple M5 MacBook Air (MPS), ~35s per model

## Results

| Metric | raw | glyph | Winner |
|--------|-----|-------|--------|
| Final train loss | 5.214 | 5.108 | glyph |
| Perplexity (val) | 213.1 | 149.3 | glyph |
| **Bits/char (val)** | 2.574 | **2.534** | glyph |
| Tokens/sec | 105.2 | 215.3 | glyph* |
| **Chars/sec** | 390.3 | **691.3** | glyph |
| Compression ratio | 0.334 | 0.352 | raw |
| Avg tokens/line (val) | 10.53 | 11.02 | raw |
| Whitespace-token ratio | — | 0.975 | glyph |

\* Tokens/sec inflated for glyph since each token encodes more source chars. **Chars/sec is fair comparison.**

## Key Findings

1. **Glyph wins on small corpus** — 2.534 vs 2.574 bits/char (1.6% better), 691 vs 390 chars/sec (77% faster inference).

2. **First successful result** — hypothesis holds at 1MB scale with Shakespeare corpus.

3. **Compression trade-off** — glyph slightly worse tokenization compression (0.352 vs 0.334 tokens/char), but wins on quality + speed.

## Comparison to Gothic

Shakespeare (1MB): glyph wins quality + speed.  
Gothic (8MB): raw wins quality, glyph wins speed → **hypothesis fails at scale**.

Possible causes:
- Encoder-corpus mismatch (Plover optimized for conversational English, not Victorian Gothic)
- BPE vocab starvation (180 special tokens from 2048 budget hurts more on larger, more diverse corpus)
- Training depth (may need more epochs on Gothic)
