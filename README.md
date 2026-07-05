# GlyphLM

GlyphLM is a minimal research experiment testing whether human-designed shorthand
compression — inspired by stenography systems like Gregg and Pitman — can make
language model tokenizers more efficient. A deterministic, rule-based encoder
compresses text (common words → single glyphs, common suffixes collapsed, doubled
vowels dropped) before it ever reaches a tokenizer, and two otherwise-identical tiny
GPT-2 models are trained: one on the original corpus, one on the glyph-compressed
version.

## Hypothesis

At an equal token budget, a model trained on glyph-compressed text achieves a better
BPE compression ratio and comparable (or better) perplexity and inference
tokens/second than a model trained on raw text — i.e. deterministic symbolic
compression is "free" efficiency that a small LLM can still learn from.

## How to run

```
uv venv --python 3.12 && uv pip install -r requirements.txt
python -m glyph.data && python -m glyph.tokenizer && python -m glyph.train
python -m glyph.eval
```

(Or open `GlyphLM_experiment.ipynb` locally or in Google Colab.)

## Results (first run)

Ran on an Apple M5 MacBook Air (MPS backend), full pipeline in well under a minute
of compute — a smoke test on the full `tiny_shakespeare` corpus (~114 training
steps total), not a statistically meaningful experiment yet. Raw numbers in
`results.json` (gitignored, regenerate with `python -m glyph.eval`).

| Metric | raw | glyph |
|---|---|---|
| Final train loss | 6.008 | **5.842** |
| Perplexity (val set) | 408.8 | **393.6** |
| Tokens/second (inference) | 169.9 | **289.9** |
| Compression ratio (tokens/char) | **0.333** | 0.358 |
| Whitespace-token ratio (glyph/raw) | 1.0 (by construction — see below) | |

**Interpretation:** glyph text trained to lower loss, better perplexity, and
much faster inference at this scale — but its *character-level* BPE compression
ratio was actually worse, not better, and the whitespace-token ratio is exactly
1.0. Both are explainable, not noise:

- The encoder substitutes one word for one glyph (e.g. `the` → `þ`), so it never
  merges multiple words into fewer tokens — whitespace-level token count can't
  drop by design.
- The glyph symbols (þ, ŋ, ʃ, ...) are novel and rare relative to common English
  morphemes, so at a fixed 2048-token BPE budget they don't merge into longer
  chunks as efficiently — this eats into the character-level compression gain.

**Chat example** (`python -m glyph.chat`, after training `model_glyph`):
```
you> hi
glyph> hi , 4 I to will r ñ & ' ' me . B ly his , ? my he me þ kŋ , if u thy ' ? my O in & þs of : my thy a ; I am d . Kŋ / me , ' , 4 þ son or þ kŋ , , he so ,
plain> hi , for I to will r not and ' ' me . B ly his , ? my he me the king , if u thy ' ? my O in and this of : my thy a ; I am d . King / me , ' , for the son or the king , , he so ,
```
Expected at this scale: a ~4M-param model trained for ~11 seconds on 1MB of text
produces Shakespearean-flavored word salad, not coherent replies. This confirms
the encode → generate → decode wiring works, nothing more.

## Ideas to improve the next run

- **Merge phrases, not just words**: real stenography (Gregg/Pitman) chords whole
  phrases into one symbol. Our encoder only does 1-word-to-1-glyph substitution,
  which is why the whitespace-token ratio can't move — adding common bigram/phrase
  rules (e.g. `for the` → one symbol) would actually shrink token counts.
- **Try SentencePiece Unigram instead of BPE** — tends to compress slightly better
  than BPE at small vocab sizes, which may help the glyph corpus's novel symbols
  compete better for vocab budget.
- **Train longer / on more data** — this first run was a ~1-minute smoke test to
  prove the pipeline works, not a real experiment. A proper comparison needs enough
  steps for both models to actually converge before comparing perplexity.
- **Vocab size sweep** — since the glyph corpus's compression disadvantage traces
  to novel symbols competing for a fixed 2048-slot vocab, trying a couple of
  different vocab sizes would show whether that gap closes, stays flat, or grows.

## Related work

[Training LLMs over Neurally Compressed Text](https://arxiv.org/abs/2404.03626)
(Lester et al., 2024) trains LLMs directly on text compressed with a *learned* neural
compressor (Equal-Info Windows over arithmetic coding), showing this can outperform
byte-level baselines at scale. GlyphLM takes a much simpler, fully deterministic and
human-interpretable approach — fixed regex rules rather than a learned codec — as a
lightweight complement to that direction.
