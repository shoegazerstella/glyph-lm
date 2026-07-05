# GlyphLM

GlyphLM is a minimal research experiment testing whether shorthand compression —
grounded in real stenography — can make language model tokenizers more efficient.
The encoder (`glyph/encoder.py` + `glyph/steno_dict.py`) shortens the most frequent
words and phrases in the training corpus using their real chords from
[Plover](https://github.com/openstenoproject/plover) (an open-source stenography
engine), rather than hand-picked symbols. Two otherwise-identical tiny GPT-2 models
are then trained: one on the original corpus, one on the shorthand-compressed
version. Full build/iteration history: [`docs/log.md`](docs/log.md).

## Hypothesis

At an equal token budget, a model trained on glyph-compressed text achieves a better
BPE compression ratio and comparable (or better) perplexity and inference
tokens/second than a model trained on raw text — i.e. deterministic symbolic
compression is "free" efficiency that a small LLM can still learn from. Framed more
generally: any gap this experiment finds should tell us whether it's a **fixed
vocab-budget artifact** (glyph symbols losing out to common English morphemes at a
given vocab size) **or a structural problem** with word-for-word shorthand
substitution itself — that distinction is what makes the comparison worth running,
not just the raw numbers.

## How to run

```
uv venv --python 3.12 && uv pip install -r requirements.txt
python -m glyph.data && python -m glyph.tokenizer && python -m glyph.train
python -m glyph.eval
```

(Or open `GlyphLM_experiment.ipynb` locally or in Google Colab.)

## Results (latest run)

Ran on an Apple M5 MacBook Air (MPS backend), full pipeline in under a minute of
compute — still a smoke test on the full `tiny_shakespeare` corpus (~100 training
steps total per model), not a statistically converged experiment yet. Raw numbers
in `results.json` (gitignored, regenerate with `python -m glyph.eval`). Two earlier
iterations (hand-picked glyphs, then hand-picked phrases + special tokens) are
recorded in [`docs/log.md`](docs/log.md) for comparison.

The current encoder fits a word/phrase → real-steno-chord mapping from the
training corpus itself (`glyph/steno_dict.py`, sourced from
[Plover](https://github.com/openstenoproject/plover)'s dictionary, GPLv2+):
top-150 words and top-30 phrases by frequency, each using its real, shortest
Plover chord. Those chords are also registered as BPE special tokens so they
don't have to compete for merge slots against common English morphemes.

| Metric | raw | glyph |
|---|---|---|
| Final train loss | 6.004 | **5.812** |
| Perplexity (val set) | 393.1 | **286.1** |
| Tokens/second (inference) | 256.8 | **415.0** |
| Compression ratio (tokens/char) | **0.333** | 0.352 |
| Avg tokens/line (val) | **10.91** | 11.45 |
| Whitespace-token ratio (glyph/raw) | — | **0.975** |

**Interpretation:** this is the first run where the glyph model wins on *every*
metric — lower loss, lower perplexity, faster inference — and the whitespace-token
ratio finally dropped below 1.0, meaning phrase-chording is genuinely reducing
token counts, not just character counts. The character-level compression gap also
shrank substantially versus the hand-picked-glyph version (0.352 vs. 0.454 — see
`docs/log.md`). Grounding the rules in real, frequency-ranked steno data clearly
outperformed the hand-picked rule set on every axis tested so far.

**Caveat carried forward:** perplexity here is still compared per-token across two
differently-tokenized streams, which isn't strictly apples-to-apples — a glyph
token can carry more original information than a raw token. The fair metric would
be bits-per-*original*-character (not yet implemented in `eval.py`).

**Chat example** (`python -m glyph.chat`, after training `model_glyph`):
```
you> hi
glyph> hi , 4 I to will r ñ & ' ' me . B ly his , ? my he me þ kŋ , if u thy ' ? my O in & þs of : my thy a ; I am d . Kŋ / me , ' , 4 þ son or þ kŋ , , he so ,
plain> hi , for I to will r not and ' ' me . B ly his , ? my he me the king , if u thy ' ? my O in and this of : my thy a ; I am d . King / me , ' , for the son or the king , , he so ,
```
(From an earlier hand-picked-glyph run — expected at this scale regardless of
encoder version: a ~4M-param model trained for ~11 seconds on 1MB of text produces
Shakespearean-flavored word salad, not coherent replies. This confirms the
encode → generate → decode wiring works, nothing more.)

## Ideas to improve the next run

- **Bits-per-original-character metric**: add a fairer, tokenization-independent
  comparison to `eval.py` — total cross-entropy in bits divided by the length of
  the *uncompressed* reference text, rather than per-token perplexity.
- **Try SentencePiece Unigram instead of BPE** — tends to compress slightly better
  than BPE at small vocab sizes, which may help chords compete better for vocab
  budget without needing the special-token reservation at all.
- **Train longer / on more data** — every run so far is a ~1-minute smoke test to
  prove the pipeline and each change work, not a converged experiment. A real
  comparison needs enough steps for both models to actually plateau.
- **Vocab size sweep** — since the remaining compression gap traces to a fixed
  2048-slot vocab budget, trying a few different vocab sizes would show whether
  that gap closes, stays flat, or grows as budget increases.

## Related work

[Training LLMs over Neurally Compressed Text](https://arxiv.org/abs/2404.03626)
(Lester et al., 2024) trains LLMs directly on text compressed with a *learned* neural
compressor (Equal-Info Windows over arithmetic coding), showing this can outperform
byte-level baselines at scale. GlyphLM takes a much simpler, fully deterministic and
human-interpretable approach — fixed regex rules rather than a learned codec — as a
lightweight complement to that direction.
