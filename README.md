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

## Expected results

_(fill in after running)_

| Metric | raw | glyph |
|---|---|---|
| Final train loss | | |
| Perplexity (val set) | | |
| Tokens/second (inference) | | |
| Avg tokens per line (val) | | |
| Compression ratio | | |

## Related work

[Training LLMs over Neurally Compressed Text](https://arxiv.org/abs/2404.03626)
(Lester et al., 2024) trains LLMs directly on text compressed with a *learned* neural
compressor (Equal-Info Windows over arithmetic coding), showing this can outperform
byte-level baselines at scale. GlyphLM takes a much simpler, fully deterministic and
human-interpretable approach — fixed regex rules rather than a learned codec — as a
lightweight complement to that direction.
