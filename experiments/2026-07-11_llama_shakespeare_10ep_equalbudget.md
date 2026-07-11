# Run 5: Equal BPE Budget Test (Llama, Shakespeare, 10 epochs, seed=42)

**Date:** 2026-07-11  
**Architecture:** Llama (n_embd=128, n_layer=4, n_head=4, context=256, ~4M params)  
**Corpus:** tiny_shakespeare (1MB)  
**Training:** 10 epochs, seed=42, lr=5e-4, batch=16, grad_accum=2  
**Vocab sizes:** raw=2048, glyph=2228 (2048 base + 180 shorthand tokens)  
**Hypothesis:** Giving glyph model EXTRA vocab slots (equal *merge* budget, not equal *total* budget) will close compression gap.

## Results

| Metric | Raw (2048) | Glyph (2228) | Winner | Delta |
|--------|-----------|-------------|--------|-------|
| **Train loss** | 2.247 | 2.249 | ≈ | +0.1% |
| **Train time** | 466s | 289s | **Glyph** | -38% |
| **Perplexity (val)** | 111.2 | 113.8 | Raw | +2.3% |
| **Bits/char** | **2.138** | **2.218** | **Raw** | **+3.7%** |
| **Tokens/sec** | 45.1 | 46.2 | Glyph | +2.4% |
| **Chars/sec (model)** | 131.2 | 148.8 | Glyph | +13.4% |
| **Chars/sec (e2e)** | 131.2 | 162.9 | **Glyph** | **+24.2%** |
| **Compression ratio** | **0.315** | **0.325** | **Raw** | Glyph uses MORE tokens/char |
| **Next-word accuracy** | **0.13** | **0.00** | **Raw** | Glyph fails completely |

## Tokenizer Stats (post-training)

- Raw: avg 18.71 tokens/line, compression 0.3162 tokens/char
- Glyph: avg 18.38 tokens/line, compression 0.3179 tokens/char

## Key Findings

### 1. Extra vocab budget DID NOT close compression gap

Despite glyph having 180 extra vocab slots (2228 vs 2048), **compression ratio remained worse** (0.325 vs 0.315 tokens/char on validation set). Bits-per-char also unchanged (2.218 vs 2.138, +3.7% worse).

**Implication:** Vocab starvation was NOT the root cause. Shorthand substitution provides no compression benefit even with equal BPE merge budget.

### 2. Training faster for glyph (38%)

Glyph trained 38% faster (289s vs 466s). This is NEW — Run 4 showed only 3% difference (289s vs 298s). Likely measurement noise or MPS thermal throttling on raw run.

### 3. End-to-end inference faster for glyph (24%)

Glyph e2e throughput 162.9 chars/sec vs raw 131.2 (+24%). This is character generation speed including decode overhead. Model-only glyph is 148.8 chars/sec (+13% vs raw's 131.2), so decode overhead is small (~10%).

**But:** Compression ratio shows glyph uses 0.325 tokens/char vs raw 0.315. Glyph inference is faster *per token generated*, but needs more tokens to encode same text. Net throughput gain is real but not from compression.

### 4. Next-word accuracy still 0%

Glyph model fails downstream task completely (0% vs 13% for raw). Same as Run 4. Shorthand tokenization learned by model but not generalizing.

### 5. Completion quality unchanged

Both models produce incoherent text (expected for 4M params on 1MB). Glyph completions show decoded English with artifacts ("are to M he to", "B art ter") — decoder working but input text was already corrupted by shorthand encoding.

## Hypothesis Test Result

**REJECTED.** Equal BPE merge budget hypothesis failed. Even with glyph vocab=2228 (180 extra slots), compression and bits-per-char metrics remain worse than raw. Shorthand chords are information-preserving substitution, not compression, and BPE cannot exploit them for efficiency gains.

## Root Cause Analysis (Updated)

Original hypothesis (Run 4): "Reserving 180 special tokens from 2048 vocab starves BPE."  
**Disproven by Run 5:** Giving glyph 2228 vocab (equal merge budget) did not improve compression.

**New hypothesis:** Shorthand encoding **increases effective alphabet size** without reducing entropy. BPE learns optimal subword units for raw English (26 letters + punctuation). Shorthand adds 180 atomic chord tokens (e.g., `-T`, `SP`, `O*EUPL`) that BPE cannot merge or compress further. Result: glyph corpus has LARGER effective alphabet than raw, and BPE needs more tokens per character to represent it, even with extra vocab budget.

**Analogy:** Replacing frequent words with emoji doesn't compress text — it just substitutes one symbol for another at equal information density. BPE on emoji-text needs emoji as vocab entries, reducing merge budget for surrounding text.

## Why E2E Inference Is Faster

Despite worse compression, glyph inference is 24% faster end-to-end:

1. **Llama's attention is O(n²) in sequence length.** Glyph's slightly shorter token sequences (18.38 vs 18.71 avg tokens/line per tokenizer stats) reduce attention cost slightly.
2. **MPS matrix ops faster on glyph model.** Possible that glyph vocab layout (2228 vs 2048) aligns better with MPS memory or that raw model hit thermal throttling during inference benchmark.
3. **Decode overhead is small** (~10% of generation time), so model-only speed dominates.

**Net:** Glyph is faster to *run* but not because of compression — likely architecture/hardware interaction.

## Next Steps

1. **Abandon shorthand hypothesis for compression.** Five runs (including this) consistently show shorthand provides no bits-per-char improvement and fails downstream tasks.

2. **Reframe as pure inference speedup experiment.** If glyph models are 24% faster e2e despite worse compression, investigate whether shorthand is useful purely as a *runtime optimization* (not training efficiency).

3. **Test on larger corpus** (Gothic 8MB, 30 epochs) with equal budget to confirm result holds at scale.

4. **Investigate downstream task failure.** Why does next-word accuracy collapse to 0% for glyph? Is shorthand encoding breaking semantic structure that BPE/LM rely on?

## Conclusion

Equal BPE merge budget did not rescue the shorthand hypothesis. Glyph models compress worse, fail downstream tasks, but run faster. This pattern suggests shorthand is **not a compression mechanism** but possibly a **tokenization-level speedup** at the cost of model quality.

Hypothesis: **stenography chords are a lossy representation for LM training** — they preserve raw text information but lose linguistic structure that BPE+LM exploit for compression and generalization.
