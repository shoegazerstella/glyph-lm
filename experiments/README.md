# Experiment Log

Each file documents one experimental run with architecture, hyperparameters, results, and analysis.

## Index

- **2026-06-XX_gpt2_shakespeare_10ep.md** — GPT-2 on Shakespeare (1MB), 10 epochs. **Glyph wins** 2.534 vs 2.574 bits/char, 77% faster inference. First successful result.
- **2026-06-XX_gpt2_gothic_30ep.md** — GPT-2 on Gothic (8MB), 30 epochs. **Raw wins quality** 1.966 vs 2.043 bits/char, glyph 48% faster. Hypothesis fails at scale. Chat shows glyph more coherent despite worse perplexity.
- **2026-07-11_llama_gothic_30ep.md** — Llama on Gothic (8MB), 30 epochs, seed=42. Llama worse than GPT-2 (2.026 vs 1.966 bits/char). E2E speed measured.
- **2026-07-11_llama_shakespeare_10ep_equalbudget.md** — Llama on Shakespeare (1MB), 10 epochs, seed=42. **Equal BPE merge budget test** (raw=2048, glyph=2228). **Raw wins compression** 2.138 vs 2.218 bits/char, **glyph 24% faster e2e** but fails downstream task (0% vs 13% accuracy). **Hypothesis rejected:** extra vocab did not close compression gap. Shorthand is NOT compression.

## Format

Each experiment log should include:
- Date and configuration (architecture, corpus, epochs, seed)
- Results table with raw vs glyph comparison
- Key findings and interpretation
- Hypotheses for unexpected results
- Next steps
