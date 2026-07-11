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

## Run 4 — 10-epoch full training (seed=42, Llama arch)

Switched architecture from GPT-2 to Llama (n_embd=128, n_layer=4, n_head=4, 
context=256) for consistency with modern small-model defaults. Trained 10 epochs 
(~46 min on M5 MPS) to approach convergence. Implemented bits-per-char metric 
(cross-entropy in bits ÷ uncompressed char count) for fair comparison across 
tokenizations. Added next-word accuracy downstream task (100-sample probe).

Results:

| Metric | raw | glyph | Winner |
|---|---|---|---|
| Final train loss | 2.247 | 2.249 | ≈ |
| Train time | 298s | 289s | glyph (3%) |
| Perplexity (val) | 111.2 | 113.8 | raw (2.3%) |
| **Bits/char** | **2.138** | **2.218** | **raw (3.7%)** |
| Tokens/sec | 30.2 | 31.1 | glyph |
| Chars/sec (model) | 88.0 | 100.1 | glyph (14%) |
| Chars/sec (e2e) | 88.0 | 104.2 | glyph (18%) |
| Compression ratio | 0.315 | 0.325 | raw (worse = fewer tokens/char) |
| **Next-word accuracy** | **0.11** | **0.00** | **raw** |

**Hypothesis REJECTED.** At 10 epochs with equal vocab budget (2048 BPE tokens):

1. **Bits-per-char worse for glyph** (2.218 vs 2.138) — shorthand provides no 
   information compression benefit. Glyph corpus carries same entropy as raw text 
   but BPE encodes it less efficiently.
2. **Next-word accuracy collapsed** — 0% vs 11%. Glyph model fails simple downstream 
   task despite comparable training loss.
3. **Inference throughput misleading** — glyph wins chars/sec because tokens map to 
   more characters, but compression ratio shows glyph uses MORE tokens per character 
   (0.325 vs 0.315). BPE learned worse subword units.
4. **Completion quality poor for both** — 4M-param models produce incoherent text. 
   Glyph completions emit raw chord tokens (`PHE`, `RAOPL`) instead of decoded 
   English, indicating encoder/decoder not robust.

**Root cause:** Stenography chords are **information-preserving substitution**, not 
compression. Reserving 180 special tokens (150 words + 30 phrases) from 2048 vocab 
prevents BPE from learning optimal subword merges on the remaining corpus. Raw BPE 
discovers compression from data; glyph BPE is constrained by pre-imposed chords.

Validated in Run 2 (hand-picked special tokens hurt compression) and now confirmed 
at convergence: shorthand harms BPE efficiency at fixed vocab budget.

**Next steps:** Either (1) abandon shorthand hypothesis, or (2) give glyph model 
EXTRA vocab budget (e.g., 2048 base + 180 special = 2228 total) so BPE merge budget 
is comparable to raw. Current setup starves BPE.

## Open questions / caveats carried forward

- Vocab-budget interactions are load-bearing. A vocab-size sweep (e.g., raw@2048 vs 
  glyph@2228) would test whether equal *merge* budget (not equal *total* budget) 
  closes the compression gap.
- Next-word accuracy metric is a simple probe (100 samples, greedy decode). More 
  robust downstream eval needed for statistical significance.
- Multi-seed run (seeds [42, 123, 456, 789, 1024]) not yet executed — Run 4 is 
  single-seed only.
