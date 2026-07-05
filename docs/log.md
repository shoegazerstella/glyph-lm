# GlyphLM — Build & Experiment Log

## Scaffold (initial build)

Built the full project per `docs/prd.md`: `glyph/encoder.py`, `glyph/data.py`,
`glyph/tokenizer.py`, `glyph/train.py`, `glyph/eval.py`, `GlyphLM_experiment.ipynb`,
`README.md`, `requirements.txt`/`pyproject.toml`. Pushed to the existing private
GitHub repo `shoegazerstella/glyph-lm`.

Key decisions:
- Use `karpathy/tiny_shakespeare`'s native `train`/`validation` splits (train feeds
  both tokenizer + model training, validation held out strictly for eval
  perplexity) instead of a manual 90/10 re-split.
- Manual PyTorch training loop (not `transformers.Trainer`) for direct MPS control.
- Repo stays private; the notebook's Colab `git clone` cell will fail on a private
  repo without a PAT — flagged as a known limitation, not solved.

## Run 1 — first local execution (Apple M5, MPS)

Set up with `uv` (Python 3.12 venv), installed `torch`/`transformers`/`tokenizers`/
`datasets`. Hit and fixed one real bug: `datasets` no longer runs
`tiny_shakespeare`'s legacy dataset-script loader — pinned to the HF
auto-converted parquet revision instead.

Full pipeline ran in well under a minute of compute (nowhere near the 15–20 min
budget). Encoder at this point: 10 hand-picked word rules + 6 suffix rules,
hand-picked Unicode glyphs (þ, &, ŧ, w/, 4, u, r, ñ, hv, þs, ʃ, ŋ, mnt, ns, ld, ıt).

Results:

| Metric | raw | glyph |
|---|---|---|
| Final train loss | 6.008 | 5.842 |
| Perplexity (val) | 408.8 | 393.6 |
| Tokens/sec (inference) | 169.9 | 289.9 |
| Compression ratio (tokens/char) | 0.333 | 0.358 |
| Whitespace-token ratio | 1.0 (by construction) | |

Finding: glyph text trained to lower loss/perplexity and faster inference, but
character-level BPE compression was actually *worse*. Explained by: (1) the
encoder substitutes 1 word → 1 glyph, never merging words, so whitespace-token
count can't drop by design; (2) novel glyph symbols are rare, so they lose the
frequency-based merge competition against common English morphemes at a fixed
2048-token vocab budget.

Added `glyph/chat.py`: an interactive REPL (encode input → generate from
`model_glyph` → decode output back to English) confirming the encode/generate/
decode wiring works end-to-end.

## Run 2 — phrase-chording + special-token pre-registration (still hand-picked)

Two changes tested together:
1. Added 10 hand-picked phrase rules (`for the`, `of the`, `in the`, `to the`,
   `and the`, `with the`, `I am`, `thou art`, `my lord`, `the king`), encoded as
   Private Use Area Unicode codepoints (U+E000+), applied *before* the
   single-word rules so phrases get chorded as a whole.
2. Registered all glyph symbols (word + phrase) as BPE trainer `special_tokens`,
   so they're guaranteed atomic vocab entries instead of competing for merge slots.

Results:

| Metric | raw | glyph |
|---|---|---|
| Final train loss | 6.089 | 4.850 |
| Perplexity (val) | 405.2 | 143.8 |
| Tokens/sec (inference) | 215.9 | 283.0 |
| Compression ratio (tokens/char) | 0.333 | 0.454 (worse) |
| Avg tokens/line | 10.91 | 13.77 (worse) |
| Whitespace-token ratio | — | 0.988 |

Finding: perplexity improved dramatically (phrase chords make common spans
trivially predictable as single tokens), but compression got *worse* — reserving
~26 special-token slots ate into the ~2048 budget available for BPE to learn
merges elsewhere, pushing the rest of the text toward shorter/character-level
tokens. This directly validates the "fixed vocab-budget artifact vs structural
problem" framing later added to the README's Hypothesis section.

## Encoder overhaul — grounding in real Plover steno chords

User feedback: a hand-picked regex list of 10 words + 10 phrases is arbitrary,
with no real basis for why those specific ones. Decision: rebuild the shorthand
mapping from **Plover** (github.com/openstenoproject/plover, GPLv2+), an
open-source real-time stenography engine, instead of guessing.

- `glyph/steno_dict.py`: downloads Plover's `main.json` (~147K chord→translation
  entries), filters to clean plain word/short-phrase translations (drops
  multi-stroke chords containing `/`, and Plover's non-text formatting "briefs"
  like `{^s}`), and for each distinct word/phrase picks its shortest available
  chord as canonical.
- `build_shorthand_map(corpus_text)` ranks these real chords by actual frequency
  in the training corpus, returning separate word and phrase maps (top 150 words,
  top 30 phrases by default). Confirmed real, substantial coverage in
  `tiny_shakespeare`: **4,956 words and 703 phrases** have genuine Plover chords —
  e.g. "the"→`-T` (freq 5719), "and"→`SP`, "of the"→`-FT`, "i am"→`O*EUPL`.
- `glyph/encoder.py` rewritten around a `ShorthandCodec`: single-pass regex
  substitution (phrases before words, longest-first) instead of a hardcoded rule
  table; `fit(corpus_text)` builds and caches the mapping to
  `data/glyph/shorthand_map.json`, `encode()`/`decode()` load the cached fit.
  `get_special_tokens()` exposes the chord list for `tokenizer.py` to register as
  BPE special tokens (same mechanism as Run 2, now driven by real data).
- Data not committed to the repo: Plover's dictionary is GPLv2+, kept out of this
  project's own license via `data/steno_cache/` in `.gitignore` — fetched at
  build time instead.

## Run 3 — real steno chords (phrase-chording + special tokens, both from real data)

Whitespace-token ratio dropped **below 1.0 for the first time** (0.9753) — genuine
evidence the phrase chords are reducing token counts now, not just characters.

Results:

| Metric | raw | glyph |
|---|---|---|
| Final train loss | 6.004 | 5.812 |
| Perplexity (val) | 393.1 | 286.1 |
| Tokens/sec (inference) | 256.8 | 415.0 |
| Compression ratio (tokens/char) | 0.333 | 0.352 |
| Avg tokens/line (val) | 10.91 | 11.45 |
| Whitespace-token ratio | — | 0.975 |

Finding: the glyph model now wins on **every** metric — lower loss, lower
perplexity, faster inference — and the character-level compression gap shrank a
lot (0.352 vs the hand-picked version's 0.454) without needing to give up the
special-token registration. Real, frequency-ranked steno data outperforms the
hand-picked rule set on every axis tested so far.

## Open questions / caveats carried forward

- All three runs are ~11-second, ~100-step smoke tests on a 1MB corpus — they
  prove the pipeline and each change work, not a converged, statistically
  significant result.
- Perplexity is still being compared per-token across two differently-tokenized
  streams, which isn't strictly apples-to-apples (a glyph token can carry more
  original information than a raw token). The fair metric is bits-per-original-
  character (total cross-entropy in bits ÷ length of the *uncompressed* reference
  text) — not yet implemented in `eval.py`.
- Vocab-budget interactions (special tokens vs BPE merge slots) are clearly load-
  bearing; a vocab-size sweep would show whether the remaining compression gap
  closes as budget grows.
