# Experiment Log

Each file documents one experimental run with architecture, hyperparameters, results, and analysis.

## Index

- **2026-06-XX_gpt2_shakespeare_10ep.md** — GPT-2 on Shakespeare (1MB), 10 epochs. **Glyph wins** 2.534 vs 2.574 bits/char, 77% faster inference. First successful result.
- **2026-06-XX_gpt2_gothic_30ep.md** — GPT-2 on Gothic (8MB), 30 epochs. **Raw wins quality** 1.966 vs 2.043 bits/char, glyph 48% faster. Hypothesis fails at scale. Chat shows glyph more coherent despite worse perplexity.
- **2026-07-11_llama_gothic_30ep.md** — Llama on Gothic (8MB), 30 epochs, seed=42. Llama worse than GPT-2 (2.026 vs 1.966 bits/char). E2E speed measured.

## Format

Each experiment log should include:
- Date and configuration (architecture, corpus, epochs, seed)
- Results table with raw vs glyph comparison
- Key findings and interpretation
- Hypotheses for unexpected results
- Next steps
