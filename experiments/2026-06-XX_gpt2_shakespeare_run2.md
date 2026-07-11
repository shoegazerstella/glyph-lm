# GPT-2 Shakespeare Run 2 (phrase-chording + special tokens)

**Date:** ~2026-06 (from docs/log.md Run 2)  
**Architecture:** GPT-2 (n_embd=128, n_layer=4, n_head=4, n_positions=256)  
**Corpus:** tiny_shakespeare (~1MB)  
**Training:** 3 epochs (smoke test)  
**Encoder:** Hand-picked 10 words + 10 phrases + 6 suffix rules, registered as BPE special_tokens

## Encoder Changes from Run 1

1. **Added 10 phrase rules** (for the, of the, in the, to the, and the, with the, I am, thou art, my lord, the king) encoded as Private Use Area Unicode (U+E000+)
2. **Phrase-before-word ordering** — phrases matched first, so multi-word phrases get chorded as a whole
3. **Special token registration** — all glyph symbols registered as BPE trainer `special_tokens` → guaranteed atomic vocab entries, don't compete for merge slots

## Results

| Metric | raw | glyph |
|--------|-----|-------|
| Final train loss | 6.089 | 4.850 |
| Perplexity (val) | 405.2 | 143.8 |
| Tokens/sec | 215.9 | 283.0 |
| Compression ratio | 0.333 | 0.454 (worse) |
| Avg tokens/line | 10.91 | 13.77 (worse) |
| Whitespace-token ratio | — | 0.988 |

## Key Findings

1. **Perplexity improved dramatically** — 143.8 vs 405.2. Phrase chords make common spans trivially predictable as single tokens.

2. **Compression got worse** — 0.454 vs Run 1's 0.358. Reserving ~26 special-token slots ate into ~2048 budget available for BPE to learn merges elsewhere → rest of text pushed toward shorter/character-level tokens.

3. **Validates "vocab-budget artifact" framing** — special token registration fixed perplexity but hurt compression. Not a structural problem with shorthand, but a fixed-vocab-budget constraint.

4. **Still hand-picked** — arbitrary choice of which 10 phrases, no corpus-frequency grounding.

## Lessons

- Special token registration fixes perplexity (prevents merge competition)
- But reserving vocab slots starves BPE elsewhere → compression regression
- Need real, frequency-ranked steno chords from Plover instead of hand-picked guesses
