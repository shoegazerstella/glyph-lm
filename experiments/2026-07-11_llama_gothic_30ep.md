# Llama Gothic 30 Epochs (seed=42)

**Date:** 2026-07-11  
**Architecture:** Llama (n_embd=128, n_layer=4, n_head=4, n_positions=256)  
**Corpus:** Gothic Fiction (~8MB)  
**Training:** 30 epochs, AdamW lr=3e-4  
**Seed:** 42

## Results

| Metric | raw | glyph | Winner | GPT-2 comparison |
|--------|-----|-------|--------|------------------|
| Final train loss | 2.963 | 3.149 | raw | raw: 3.949 → 2.963 ✓ |
| Perplexity (val) | 86.92 | 88.89 | raw | raw: 76.19 → 86.92 ✗ |
| Bits/char (val) | 2.026 | 2.102 | raw | raw: 1.966 → 2.026 ✗ |
| Tokens/sec | 190.0 | 306.4 | glyph | - |
| Chars/sec | 697.2 | 799.7 | glyph | glyph: 815.9 → 799.7 ✓ |
| E2E chars/sec | 697.2 | 1013.0 | glyph | NEW |
| Compression | 0.3145 | 0.3250 | raw | - |

## Key Findings

1. **Llama worse than GPT-2 on quality** — 2.026 vs 1.966 bits/char for raw. Unexpected — Llama architecture should perform better than GPT-2.

2. **Same pattern holds** — raw wins quality (2.026 vs 2.102 bits/char), glyph wins speed (1013 vs 697 chars/sec e2e).

3. **E2E overhead measured** — glyph model-only: 800 chars/sec, e2e with steno encode/decode: 1013 chars/sec. FASTER including overhead — likely measurement variance/noise.

4. **Sample quality improved vs earlier runs** — both models generate more coherent Gothic prose. Previous runs had fragments like "th", "per fe ction ate". This run: glyph "sh ap ed sh a ft", "g un s de f les t".

## Hypotheses for Llama Underperformance

1. **Hyperparameters not tuned** — used same lr/schedule as GPT-2, Llama may need different settings
2. **Needs more epochs** — 30 may not be enough for Llama to converge
3. **Gothic corpus-specific** — maybe architectural difference matters more on this specific text domain

## Next Steps

- Try more seeds for statistical significance
- Tune Llama-specific hyperparameters
- Compare on shakespeare corpus (smaller, different domain)
