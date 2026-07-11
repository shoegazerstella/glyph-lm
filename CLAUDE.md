# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GlyphLM is a controlled experiment testing whether steno-chord shorthand compression makes small language models more efficient to train and run. Two identical tiny GPT-2s are trained on the same corpus: one on raw text, one on text where frequent words/phrases are replaced by real Plover stenography chords before tokenization. Everything else (architecture, vocab size, hyperparameters) is held constant.

**Core hypothesis:** At equal vocab budget, shorthand-compressed text yields better tokens-per-character compression, comparable or better perplexity, and faster inference.

## Running the Experiment

Full pipeline (data prep → tokenizer training → model training → eval):
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
python -m glyph.data && python -m glyph.tokenizer && python -m glyph.train
python -m glyph.eval
```

Or open `GlyphLM_experiment.ipynb` locally or in Colab.

Interactive chat with trained shorthand model:
```bash
python -m glyph.chat
```

## Architecture

The pipeline is sequential — each step depends on outputs from the prior step:

1. **`glyph.steno_dict`** — Downloads Plover's real steno chord dictionary, filters to plain word/phrase translations, ranks by corpus frequency
2. **`glyph.encoder`** — Fits a word/phrase → chord mapping from training corpus (top 150 words, top 30 phrases by frequency), then substitutes via single-pass regex. Call `fit()` once during data prep; `encode()`/`decode()` use the cached mapping from `data/glyph/shorthand_map.json`
3. **`glyph.data`** — Downloads corpus (Gothic Fiction ~8MB or tiny_shakespeare ~1MB via `python -m glyph.data [gothic|shakespeare]`), produces raw and glyph-encoded corpora, holds out a validation split (never used for training). `train` split is used for both tokenizer training and model training; `validation` is held strictly for eval.py perplexity
4. **`glyph.tokenizer`** — Trains two independent BPE tokenizers (vocab_size=2048), one per corpus. Shorthand chords are registered as BPE special tokens so they don't compete for merge slots against English subwords — this was critical in practice (see README §6)
5. **`glyph.train`** — Trains two identical models (GPT-2 or Llama architecture via `ARCHITECTURE` constant, n_embd=128, n_layer=4, n_head=4, n_positions=256), 30 epochs, AdamW lr=3e-4. Supports seed arg for reproducibility (`python -m glyph.train [seed]`). Auto-detects MPS/CUDA/CPU. Concatenates corpus into one token stream and chunks into fixed-size blocks rather than padding per-line (keeps token throughput comparable across raw vs glyph)
6. **`glyph.eval`** — Computes held-out perplexity, bits-per-char (tokenization-normalized metric), inference tokens/sec and chars/sec (both model-only and end-to-end with steno overhead), compression ratio, sample completions. Outputs `results.json` (gitignored)
7. **`glyph.chat_glyph`** — Interactive REPL for glyph model: input shorthand-encoded → generated → decoded back to English (`python -m glyph.chat_glyph`)
8. **`glyph.chat_raw`** — Interactive REPL for raw model: plain English in/out (`python -m glyph.chat_raw`)
9. **`run_multi_seed.py`** — Runs training + eval across multiple seeds (default [42, 123, 456, 789, 1024]), archives per-seed results, computes mean ± std for statistical significance

## Key Constraints

- **Vocab budget is the real constraint, not shorthand itself.** Iteration history (`docs/log.md`) shows how hand-picked symbols hurt BPE compression by losing the frequency-based merge competition, and how reserving special tokens without frequency-ranking starved BPE elsewhere. Switching to real, frequency-ranked Plover chords fixed both.
- **Perplexity is compared per-token across two different tokenizations** — not strictly apples-to-apples since a shorthand token can carry more original information than a raw token. Correct normalization is bits-per-original-character (total cross-entropy ÷ uncompressed text length); now implemented in eval.py as `bits_per_char`.
- **Training depth increased to 10 epochs** — earlier smoke tests (~100 steps, 3 epochs) showed mechanism works; 10 epochs moves closer to convergence. Multi-seed runner (`run_multi_seed.py`) computes statistical significance via mean ± std across seeds.

## Metric Definitions

All computed in `glyph/eval.py`:

- **Perplexity** — `exp(mean cross-entropy loss)` on held-out validation split, computed per-token in each model's own tokenization (not directly comparable across tokenizations)
- **Bits-per-char** — Total cross-entropy loss in bits ÷ original character count of uncompressed text. Fair comparison metric across tokenizations since it normalizes by source text length, not token count
- **Compression ratio** — BPE tokens ÷ character count of the (raw or shorthand) text
- **Whitespace-token ratio** — glyph corpus word count ÷ raw corpus word count via plain whitespace split. Only metric directly comparable across corpora regardless of tokenizer (doesn't depend on BPE)
- **Tokens/second** — `model.generate()` throughput on a fixed prompt, greedy decoding (inflated for glyph model since tokens carry more chars)
- **Chars/second** — Character throughput after decoding generated tokens. Fair comparison for inference speed across tokenizations

## Dependencies

Python 3.12 (pinned in `pyproject.toml` — 3.13 is explicitly excluded). Core deps:
- `torch>=2.0` — MPS/CUDA/CPU auto-detect in train.py
- `transformers>=4.40` — GPT2LMHeadModel, PreTrainedTokenizerFast
- `tokenizers>=0.19` — BPE training
- `datasets>=2.18` — loads `karpathy/tiny_shakespeare` (pinned to refs/convert/parquet revision to avoid legacy dataset script issues)

## Working with the Encoder

The fitted shorthand mapping is cached to `data/glyph/shorthand_map.json` after running `python -m glyph.data`. If you regenerate the corpus or change the frequency-ranking logic in `steno_dict.py`, delete this file to force a refitting pass.

Chord boundary regex uses `[A-Za-z0-9*-]` lookarounds instead of `\b` since chords start/end with '-' or '*' (non-word characters that break `\b`).

Encode/decode are single-pass regex subs: phrases before words, longest-first, so multi-word phrases get chorded as a whole instead of consumed word-by-word. This is the only way `encode()` reduces whitespace-level token count.

## Build/Iteration History

Full log of every encoder version tried (not just the current one) is in `docs/log.md`. Each iteration shows why vocab budget constraints mattered more than the compression mechanism itself.
