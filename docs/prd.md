# GlyphLM — Claude Code Prompt

## Project Overview

Build **GlyphLM**: a minimal research experiment that trains a small language model on
human-designed shorthand-compressed text, then compares it against a baseline trained
on the original corpus. The goal is to measure whether symbolic compression (inspired
by stenography systems like Gregg/Pitman) produces a more efficient tokenizer and
whether a model trained on compressed text achieves comparable perplexity to one
trained on raw text, at equal token budget.

---

## What to Build

### 1. Shorthand Encoder (`glyph/encoder.py`)

A deterministic, rule-based text compressor. Apply these transformations in order:

```python
RULES = {
    # High-frequency words → single ASCII glyph
    r'\bthe\b':   'þ',
    r'\band\b':   '&',
    r'\bthat\b':  'ŧ',
    r'\bwith\b':  'w/',
    r'\bfor\b':   '4',
    r'\byou\b':   'u',
    r'\bare\b':   'r',
    r'\bnot\b':   'ñ',
    r'\bhave\b':  'hv',
    r'\bthis\b':  'þs',
    # Suffixes
    r'tion\b':    'ʃ',
    r'ing\b':     'ŋ',
    r'ment\b':    'mnt',
    r'ness\b':    'ns',
    r'ould\b':    'ld',
    r'ight\b':    'ıt',
    # Drop double vowels (keep first)
    r'([aeiou])\1+': r'\1',
}
```

Implement `encode(text: str) -> str` and `decode(text: str) -> str` (decode is
best-effort / approximate — that's fine for this experiment).

---

### 2. Data Pipeline (`glyph/data.py`)

- Load `karpathy/tiny_shakespeare` from HuggingFace datasets
- Produce two versions of the corpus:
  - `data/raw/train.txt` — original text
  - `data/glyph/train.txt` — shorthand-encoded text
- Compute and print **compression ratio**: `len(glyph_tokens) / len(raw_tokens)`
  using a simple whitespace tokenizer first, then again after BPE training

---

### 3. Tokenizer Training (`glyph/tokenizer.py`)

Train two BPE tokenizers using HuggingFace `tokenizers` library:

- `tokenizers/raw_bpe/` — trained on `data/raw/train.txt`, vocab_size=2048
- `tokenizers/glyph_bpe/` — trained on `data/glyph/train.txt`, vocab_size=2048

Print for each:
- Vocabulary size
- Average tokens per line
- Compression ratio vs character count

---

### 4. Model Training (`glyph/train.py`)

Train two identical tiny GPT-2 models using the `transformers` library:

```python
config = GPT2Config(
    vocab_size=2048,
    n_positions=256,
    n_embd=128,
    n_layer=4,
    n_head=4,
)
```

- `model_raw/` — trained on raw text with raw BPE tokenizer
- `model_glyph/` — trained on glyph text with glyph BPE tokenizer

Training: 3 epochs, batch_size=32, AdamW lr=3e-4. Log train loss every 100 steps.
Target: runs in under 15 minutes on Apple M-series (MPS backend) or Colab free T4.

Use `torch.device("mps")` if available, else `cuda`, else `cpu`.

---

### 5. Evaluation (`glyph/eval.py`)

Compare the two models:

| Metric | raw model | glyph model |
|--------|-----------|-------------|
| Final train loss | | |
| Perplexity (val set) | | |
| Tokens/second (inference) | | |
| Avg tokens per sentence | | |
| Compression ratio | | |

Print results as a formatted table to stdout.
Also generate 3 sample completions from each model given the same seed prompt.

---

### 6. Notebook (`GlyphLM_experiment.ipynb`)

Wrap the full pipeline in a single Jupyter notebook with:
- One cell per step (encode → tokenize → train → eval)
- Markdown cells explaining what each step measures and why
- A final "Results & Interpretation" section with placeholder text for findings

The notebook must run end-to-end on **Google Colab free tier** (T4, ~12GB VRAM)
with a single `!pip install` cell at the top.

---

## Project Structure

```
GlyphLM/
├── glyph/
│   ├── __init__.py
│   ├── encoder.py        # shorthand rules
│   ├── data.py           # corpus download & encoding
│   ├── tokenizer.py      # BPE training
│   ├── train.py          # model training loop
│   └── eval.py           # comparison metrics
├── data/
│   ├── raw/
│   └── glyph/
├── tokenizers/
│   ├── raw_bpe/
│   └── glyph_bpe/
├── models/
│   ├── model_raw/
│   └── model_glyph/
├── GlyphLM_experiment.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements (`requirements.txt`)

```
torch>=2.0
transformers>=4.40
tokenizers>=0.19
datasets>=2.18
numpy
jupyter
```

---

## README

Write a concise README covering:
- What GlyphLM is (one paragraph)
- The hypothesis being tested
- How to run (3 commands max)
- Expected results section (to be filled after experiment)
- Link to the "Training LLMs over Neurally Compressed Text" paper (arxiv 2404.03626)
  as related work

---

## Constraints & Notes

- Keep everything **self-contained**: no external APIs, no paid services
- The full experiment (data + train + eval) must complete in **≤20 min on M-series
  or Colab free**
- Do not use `tiny_shakespeare` for the tokenizer *and* the model training on the
  same split — use 90/10 train/val split
- Prefer readability over cleverness: this is a research prototype, not production code
- Add a `# GLYPH NOTE:` comment wherever a design decision was made that affects
  the compression/learnability tradeoff
