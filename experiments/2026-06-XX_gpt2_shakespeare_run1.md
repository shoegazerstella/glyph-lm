# GPT-2 Shakespeare Run 1 (initial smoke test)

**Date:** ~2026-06 (from docs/log.md Run 1)  
**Architecture:** GPT-2 (n_embd=128, n_layer=4, n_head=4, n_positions=256)  
**Corpus:** tiny_shakespeare (~1MB)  
**Training:** 3 epochs (smoke test)  
**Encoder:** Hand-picked 10 words + 6 suffix rules, novel Unicode glyphs

## Encoder Version

- 10 hand-picked word rules (þ=the, &=and, ŧ=that, w/=with, 4=for, u=you, r=are, ñ=not, hv=have, þs=this)
- 6 suffix rules (ʃ=tion, ŋ=ing, mnt=ment, ns=ness, ld=ould, ıt=ight)
- 1-word → 1-glyph substitution (never merges words)

## Results

| Metric | raw | glyph |
|--------|-----|-------|
| Final train loss | 6.008 | 5.842 |
| Perplexity (val) | 408.8 | 393.6 |
| Tokens/sec | 169.9 | 289.9 |
| Compression ratio | 0.333 | 0.358 |
| Whitespace-token ratio | 1.0 (by design) | - |

## Key Findings

1. **Glyph wins on perplexity/speed** — lower loss (5.842 vs 6.008), faster inference (289.9 vs 169.9 tok/sec)

2. **Compression worse** — 0.358 vs 0.333 tokens/char. Novel glyph symbols rare → lose frequency-based merge competition against common English morphemes at fixed 2048 vocab budget.

3. **Whitespace-token count can't drop** — encoder substitutes 1 word → 1 glyph, never merges words. By design, no phrase reduction.

4. **Smoke test only** — ~11 seconds training, ~100 steps. Proves pipeline works, not converged result.

## Lessons

- Novel symbols hurt BPE compression due to vocab budget competition
- Need phrase rules to reduce whitespace-token count
- Hand-picked word list arbitrary
