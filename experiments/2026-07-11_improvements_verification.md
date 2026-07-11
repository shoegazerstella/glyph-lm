# Improvements Verification: SentencePiece Unigram & Llama (seed=42)

**Date:** 2026-07-11  
**Architecture:** Llama (n_embd=512, n_layer=8, n_head=8, vocab_size=2048)  
**Tokenizer:** SentencePiece Unigram  
**Corpus:** TODO (shakespeare or gothic)  
**Training:** 12 epochs, AdamW lr=3e-4  
**Seed:** 42

## Results

| Metric | raw | glyph | Winner | Notes |
|--------|-----|-------|--------|-------|
| Final train loss | TODO | TODO | | |
| Perplexity (val) | TODO | TODO | | |
| Bits/char (val) | TODO | TODO | | Fair comparison across tokenizations |
| Next-word accuracy (100 samples) | TODO | TODO | | Downstream task: 32-word context |
| Tokens/sec | TODO | TODO | | Model-only throughput |
| Chars/sec | TODO | TODO | | After token decoding |
| E2E chars/sec | TODO | TODO | | Including steno encode/decode overhead |
| Compression ratio | TODO | TODO | | BPE tokens ÷ char count |
| Whitespace-token ratio | TODO | TODO | | glyph words ÷ raw words |

## Success Criteria Checklist

- [ ] Unigram tokenizer successfully trains without errors
- [ ] Both raw and glyph models reach convergence (train loss < baseline)
- [ ] Bits-per-char is computed correctly (normalized by original char count)
- [ ] Next-word accuracy evaluation runs without errors
- [ ] Inference speed metrics (tokens/sec, chars/sec) are reasonable
- [ ] E2E speed includes full steno encode/decode pipeline
- [ ] Sample completions are coherent

## Key Findings

### SentencePiece Unigram Tokenizer

1. **Compression improvement:** TODO — note expected bits-per-char vs BPE baseline
2. **Frequency ranking:** TODO — did unigram better preserve chord priorities?
3. **OOV behavior:** TODO — any differences in unknown token handling?

### Downstream Task (Next-Word Accuracy)

1. **Task setup:** 100 random validation samples, 32-word context window
2. **Raw model accuracy:** TODO %
3. **Glyph model accuracy:** TODO %
4. **Interpretation:** TODO — which model generalizes better to word prediction?

### Speed and Compression

1. **E2E overhead:** TODO — encode/decode overhead impact on total throughput
2. **Compression vs baseline:** TODO — compare to BPE results from earlier runs
3. **Quality-speed tradeoff:** TODO — does unigram shift the frontier?

## Hypotheses for This Run

- Unigram tokenizer with frequency ranking should outperform BPE at fixed vocab size
- Larger Llama (512/8/8) should show clearer quality advantage over GPT-2
- 12 epochs may be optimal convergence point (was 10 before, 30 for gothic)
- Next-word accuracy may reveal generalization differences invisible in perplexity

## Next Steps

- Multi-seed verification (seeds: 42, 123, 456, 789, 1024)
- Compare bits-per-char across tokenizer types (BPE, Unigram) at fixed vocab
- Document final results in main README
- Archive per-seed results for reproducibility
