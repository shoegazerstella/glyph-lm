# GlyphLM Improvements: Conservative Upgrade

**Date:** 2026-07-11  
**Status:** Design  
**Goal:** Improve tokenization, architecture, and evaluation to close quality gap and validate hypothesis at scale

## Background

Current state (from experiments/2026-07-11_llama_gothic_30ep.md):
- Llama (128 hidden, 4 layers) underperforms GPT-2: 2.026 vs 1.966 bits/char on Gothic
- Glyph compression worse: 0.3250 vs 0.3145 tokens/char (3.3% gap)
- Perplexity misleading: glyph chat more coherent despite worse score
- BPE tokenizer at 4096 vocab: shorthand special tokens (180) compete for merge slots

## Objectives

1. **Quality:** Close bits/char gap (target: glyph ≤ raw + 2%)
2. **Speed:** Maintain or improve inference throughput (glyph already 45% faster e2e)
3. **Architecture:** Proper small Llama that matches/beats GPT-2 baseline
4. **Evaluation:** Add downstream task that measures generation quality, not just perplexity

## Constraints

- Training time budget: ~15-25 min per model on M1 MacBook Air (user hardware)
- Backwards compatibility: existing models/tokenizers still work
- Fair comparison: same vocab size (4096), same training data, same eval methodology

## Design

### 1. Tokenizer: BPE → SentencePiece Unigram

**Problem:** BPE greedy merge creates suboptimal splits at fixed vocab budget. Glyph special tokens waste slots that BPE can't reclaim.

**Solution:** SentencePiece Unigram LM trains probabilistic subword model via EM algorithm, prunes low-probability tokens → better compression.

**Implementation:**
- `glyph/tokenizer.py`: add `train_sentencepiece(corpus_path, output_dir, vocab_size, special_tokens)` function
- Uses `sentencepiece.SentencePieceTrainer.train()` with:
  - `model_type='unigram'`
  - `vocab_size=4096`
  - `user_defined_symbols=` list of shorthand chords from `encoder.get_special_tokens()`
  - `input=` corpus text file
  - `model_prefix=` output path for `.model` and `.vocab` files
- Wrap trained model with `tokenizers.SentencePieceUnigramTokenizer` for transformers compatibility
- Save as JSON tokenizer file (same format as BPE) to `{output_dir}/tokenizer.json`

**CLI:**
```bash
python -m glyph.tokenizer --type unigram  # new
python -m glyph.tokenizer --type bpe      # default, backwards compat
```

**Expected outcome:** Compression gap narrows from 3.3% to <2%. Unigram prunes redundant subwords more aggressively, reducing vocab waste from special tokens.

### 2. Architecture: Proper Small Llama

**Problem:** Current Llama config (128 hidden, 4 layers) too small, underperforms GPT-2.

**Solution:** Use production-viable small Llama config with ~40M params.

**Config:**
```python
LlamaConfig(
    vocab_size=tok.vocab_size,
    max_position_embeddings=256,    # unchanged
    hidden_size=512,                 # 128 → 512
    num_hidden_layers=8,             # 4 → 8
    num_attention_heads=8,           # 4 → 8
    intermediate_size=2048,          # 512 → 2048 (4x hidden for SwiGLU)
    rms_norm_eps=1e-5,
)
```

**Param count:** ~40M (vs current ~4M, production Llama 3.2 1B = ~1B)

**Training adjustments for time budget:**
- Epochs: 50 → 12 (targets ~15-25 min on M1)
- Keep: warmup 100 steps, cosine decay to 10% peak lr, grad accum (effective batch 32), weight decay 0.01
- Learning rate: 3e-4 (GPT-2 default, can tune if needed)

**Implementation:**
- `glyph/train.py`: update `ARCHITECTURE == "llama"` branch config values
- Update `EPOCHS = 12`
- No other changes needed (schedule, optimizer already improved in prior commit)

**Expected outcome:** Llama matches or beats GPT-2 baseline on bits/char. 12 epochs sufficient to show convergence trend.

### 3. Downstream Task: Next-Word Accuracy

**Problem:** Perplexity misleading (measures per-token accuracy in each tokenization, not language quality). Bits/char better but doesn't measure generation usefulness.

**Solution:** Simple downstream task: predict masked next word on held-out Gothic passages.

**Method:**
1. Sample 100 random spans from validation set (each 32 tokens context)
2. Mask final word, prompt model to predict next word (greedy decode)
3. Check if prediction matches ground truth (exact match)
4. Accuracy = % correct / 100 samples
5. For glyph model: decode predicted chord back to raw text before checking correctness

**Why this task:**
- Simple, interpretable, fast (~30 sec per model)
- Doesn't depend on tokenization (compares decoded text)
- Tests whether model learned Gothic language patterns vs memorizing training data

**Implementation:**
- New file: `glyph/downstream.py`
- Function signature:
  ```python
  def next_word_accuracy(
      model: PreTrainedModel,
      tokenizer: PreTrainedTokenizerFast,
      val_text: str,
      is_glyph: bool,
      n_samples: int = 100,
      context_len: int = 32
  ) -> float:
      """Returns accuracy as float 0.0-1.0"""
  ```
- Sampling:
  - Split `val_text` by whitespace, pick 100 random indices
  - Extract context window `[idx-context_len:idx]`, ground truth word `[idx]`
  - If glyph: encode context before tokenizing
  - Tokenize, run `model.generate(max_new_tokens=1, do_sample=False)`
  - Decode prediction, compare to ground truth (case-insensitive)
- Called from `eval.py`:
  ```python
  raw_acc = next_word_accuracy(raw_model, raw_tok, val_text, is_glyph=False)
  glyph_acc = next_word_accuracy(glyph_model, glyph_tok, val_text, is_glyph=True)
  results["next_word_accuracy"] = {"raw": raw_acc, "glyph": glyph_acc}
  ```

**Expected outcome:** If glyph accuracy ≥ raw despite worse bits/char, validates that perplexity/bits-per-char don't capture full quality picture. Aligns with qualitative observation (glyph chat more coherent).

### 4. Dependencies

**Add to requirements.txt / pyproject.toml:**
```
sentencepiece>=0.1.99
```

**Why:** SentencePiece library for Unigram tokenizer training. Not a transformers built-in (transformers only wraps pre-trained SentencePiece models).

## Files Changed

**Modified:**
- `glyph/tokenizer.py` — add `train_sentencepiece()`, add CLI `--type` arg
- `glyph/train.py` — update Llama config (512/8/8/2048), set EPOCHS=12
- `glyph/eval.py` — call `downstream.next_word_accuracy()`, add to results
- `requirements.txt` — add sentencepiece dependency

**New:**
- `glyph/downstream.py` — next-word accuracy evaluation

**Unchanged:**
- `glyph/encoder.py`, `glyph/data.py`, `glyph/steno_dict.py` — data pipeline same
- `glyph/chat_*.py` — chat REPLs work without changes
- Existing trained models — still loadable, no migration needed

## Migration Path

**For new runs (recommended):**
```bash
# Regenerate data if needed
python -m glyph.data gothic

# Train Unigram tokenizer (new)
python -m glyph.tokenizer --type unigram

# Train proper small Llama (updated config)
python -m glyph.train

# Eval with new downstream task
python -m glyph.eval
```

**For comparing BPE vs Unigram:**
```bash
# Train both tokenizers
python -m glyph.tokenizer --type bpe
mv models/model_raw models/model_raw_bpe
mv models/model_glyph models/model_glyph_bpe

python -m glyph.tokenizer --type unigram
python -m glyph.train
# Compare results.json from each run
```

**Backwards compatibility:**
- Default `--type bpe` means existing scripts/notebooks work without changes
- Existing model checkpoints load normally (config auto-detected via transformers)

## Success Criteria

**Primary:**
1. **Compression gap narrows:** glyph tokens/char within 2% of raw (currently 3.3%)
2. **Quality gap narrows:** glyph bits/char within 2% of raw (currently 3.8% worse: 2.102 vs 2.026)
3. **Llama beats GPT-2 baseline:** Llama raw bits/char ≤ GPT-2 raw (1.966 on Gothic)

**Secondary:**
4. **Downstream task validates perplexity:** if glyph next-word accuracy ≥ raw, confirms perplexity misleading
5. **Training time acceptable:** ≤25 min per model on M1 MacBook Air
6. **Inference speed maintained:** glyph chars/sec ≥ raw chars/sec (currently 1013 vs 697, 45% faster)

## Risks & Mitigations

**Risk:** SentencePiece Unigram doesn't improve compression as expected.  
**Mitigation:** Keep BPE as fallback (default CLI arg). Can compare directly.

**Risk:** Proper small Llama trains too slowly (>25 min) on M1.  
**Mitigation:** Reduce epochs further (12→8) or reduce batch size. Already using grad accum to keep memory low.

**Risk:** Next-word accuracy too noisy (high variance across samples).  
**Mitigation:** Increase n_samples to 200-500 if needed. 100 is starting point for speed.

**Risk:** 12 epochs insufficient for convergence.  
**Mitigation:** Check if loss still dropping at epoch 12. If so, document in experiment log and note "needs longer training" rather than claiming failure.

## Open Questions

None — all scope decisions made during brainstorming.

## Future Work (Out of Scope)

- Multi-seed statistical significance (can run after validating single seed)
- Human eval or LLM-as-judge for chat quality
- Additional downstream tasks (few-shot completion, perplexity on different domain)
- Vocab size sweep (1024, 2048, 4096, 8192)
- Llama hyperparameter tuning (lr, schedule, batch size from Llama paper)
- Bigger Llama configs (768 hidden, 12 layers) — exceeds time budget
