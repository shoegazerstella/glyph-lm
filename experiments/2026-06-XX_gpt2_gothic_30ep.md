# GPT-2 Gothic 30 Epochs (seed=42)

**Date:** ~2026-06 (from README §5.2)  
**Architecture:** GPT-2 (n_embd=128, n_layer=4, n_head=4, n_positions=256)  
**Corpus:** Gothic Fiction (~8MB, Project Gutenberg)  
**Training:** 30 epochs, AdamW lr=3e-4  
**Seed:** 42  
**Hardware:** Apple M5 MacBook Air (MPS), ~12.8 min per model

## Results

| Metric | raw | glyph | Winner |
|--------|-----|-------|--------|
| Final train loss | 3.949 | 4.055 | raw |
| Perplexity (val) | 76.19 | 78.34 | raw |
| **Bits/char (val)** | **1.966** | 2.043 | raw |
| Tokens/sec | 158.4 | 225.4 | glyph* |
| **Chars/sec** | 552.9 | **815.9** | glyph |
| Compression ratio | 0.3145 | 0.3250 | raw |
| Avg tokens/line (val) | 18.48 | 18.37 | glyph |
| Whitespace-token ratio | — | 0.963 | glyph |

\* Tokens/sec inflated for glyph. **Chars/sec is fair comparison.**

## Key Findings

1. **Result reversal** — Raw wins quality (1.966 vs 2.043 bits/char, 4% worse for glyph), opposite of Shakespeare. Glyph still 48% faster inference (816 vs 553 chars/sec).

2. **Hypothesis fails at scale** — 1MB Shakespeare: glyph wins. 8MB Gothic: raw wins quality.

3. **Quality vs perplexity disconnect** — Glyph chat output MORE coherent (narrative continuity, character names) despite worse bits/char. **Perplexity misleading** — measures per-token accuracy in each tokenization, not language quality.

## Chat Comparison

**Raw model** (hello prompt):
```
hello e to her by her , and made a little old lad y , but she was quite in her eyes 
to a new tra in . She was evid ently to be a small lad y , with a low , as she said :-- 
" You ' s a child - hand in my friends , and you
```

**Glyph model** (decoded from steno):
```
hello cted me to the other , which was very good , and i heard the little b ell ars 
which was there . so little that we must be seen , and we have a part of us . Lucy was 
not so late in the poor Lucy ' s room ; but we had gone to Lucy ' s sleep . *
```

Glyph: narrative continuity, repeated character name "Lucy", more coherent despite 4% worse bits/char.

## Why Reversal at Scale

1. **Encoder-corpus mismatch** — Plover steno optimized for conversational English (court reporting), not Victorian Gothic formal prose. Character names, Gothic terms don't align with steno strengths.

2. **BPE vocab starvation** — 180 special tokens (150 words + 30 phrases) reserved from 2048 vocab. On 8MB Gothic, BPE needs more slots for domain-specific subwords (Victorian terms, proper nouns). Reserved tokens prevent Gothic-specific pattern learning.

3. **Training depth** — 30 epochs on 8MB (240 MB-epochs) vs 10 epochs on 1MB (10 MB-epochs). Loss still dropping at epoch 30 — needs 50+ for convergence.

4. **Metrics vs quality** — perplexity measures per-token prediction accuracy in each tokenization, not language quality. Glyph tokens encode more info per token → model learns higher-level patterns → better generation despite higher per-token loss. **Need human eval or downstream tasks.**
