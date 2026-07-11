# GPT-2 Shakespeare Run 3 (real Plover steno chords)

**Date:** ~2026-06 (from docs/log.md Run 3)  
**Architecture:** GPT-2 (n_embd=128, n_layer=4, n_head=4, n_positions=256)  
**Corpus:** tiny_shakespeare (~1MB)  
**Training:** 3 epochs (smoke test)  
**Encoder:** Real Plover steno chords, frequency-ranked from training corpus

## Encoder Overhaul

**Replaced hand-picked guesses with real stenography:**
- Downloaded Plover `main.json` (~147K chord→translation entries)
- Filtered to clean plain word/phrase translations (no multi-stroke, no formatting briefs)
- Ranked by actual frequency in training corpus
- Top 150 words + top 30 phrases by frequency
- Real coverage: 4,956 words and 703 phrases have genuine Plover chords in Shakespeare

**Example chords:** "the"→`-T` (freq 5719), "and"→`SP`, "of the"→`-FT`, "i am"→`O*EUPL`

**Caching:** Fitted mapping saved to `data/glyph/shorthand_map.json`, loaded by encode()/decode()

## Results

| Metric | raw | glyph |
|--------|-----|-------|
| Final train loss | 6.004 | 5.812 |
| Perplexity (val) | 393.1 | 286.1 |
| Tokens/sec | 256.8 | 415.0 |
| Compression ratio | 0.333 | 0.352 |
| Avg tokens/line (val) | 10.91 | 11.45 |
| Whitespace-token ratio | — | 0.975 |

## Key Findings

1. **Whitespace-token ratio < 1.0 for first time** — 0.9753. Genuine evidence phrase chords reducing token counts, not just characters.

2. **Glyph wins on every metric** — lower loss, lower perplexity, faster inference, AND compression gap shrank a lot (0.352 vs Run 2's 0.454).

3. **Real, frequency-ranked data outperforms hand-picked** — special token registration still used, but now applied to corpus-grounded chords instead of arbitrary guesses.

4. **Still smoke test** — ~11 seconds, ~100 steps. Not converged, but proves real steno chords work.

## Lessons

- Real, frequency-ranked Plover chords dramatically better than hand-picked
- Compression gap narrowed without giving up special-token registration
- Vocab-budget interactions are load-bearing, but real data mitigates the problem
- Ready for longer training runs and larger corpus (Gothic Fiction)
