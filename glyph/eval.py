"""Compare the raw and glyph models on perplexity, speed, and sample output."""

import math
import time

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from glyph.encoder import decode, encode
from glyph.train import BLOCK_SIZE, get_device

SEED_PROMPT = "ROMEO:\nBut soft, what light"

MODELS = [
    ("raw", "models/model_raw", "data/raw/val.txt"),
    ("glyph", "models/model_glyph", "data/glyph/val.txt"),
]


def compute_perplexity(model, tok, text: str, device, block_size: int = BLOCK_SIZE) -> float:
    ids = tok.encode(text)
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(ids) - block_size, block_size):
            chunk = torch.tensor(ids[i : i + block_size]).unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk)
            losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses))


def tokens_per_second(model, tok, prompt: str, device, max_new_tokens: int = 100) -> float:
    input_ids = torch.tensor([tok.encode(prompt)]).to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - start
    n_generated = out.shape[1] - input_ids.shape[1]
    return n_generated / elapsed


def sample_completions(model, tok, prompt: str, device, n: int = 3, max_new_tokens: int = 60):
    input_ids = torch.tensor([tok.encode(prompt)]).to(device)
    completions = []
    for i in range(n):
        torch.manual_seed(i)
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, temperature=0.8
        )
        completions.append(tok.decode(out[0].tolist()))
    return completions


def main() -> None:
    device = get_device()
    results = {}

    for label, model_dir, val_path in MODELS:
        tok = PreTrainedTokenizerFast(
            tokenizer_file=f"{model_dir}/tokenizer.json", pad_token="<pad>"
        )
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

        with open(val_path) as f:
            val_text = f.read()

        # GLYPH NOTE: both models are prompted with the same underlying seed
        # prompt, encoded via encode() for the glyph model, so the comparison stays
        # fair (same semantic prompt); glyph completions are decoded back for
        # readability, acknowledging decode() is lossy (see encoder.py).
        prompt = SEED_PROMPT if label == "raw" else encode(SEED_PROMPT)
        completions = sample_completions(model, tok, prompt, device)
        if label == "glyph":
            completions = [decode(c) for c in completions]

        results[label] = {
            "perplexity": compute_perplexity(model, tok, val_text, device),
            "tokens_per_second": tokens_per_second(model, tok, prompt, device),
            "avg_tokens_per_line": _avg_tokens_per_line(tok, val_text),
            "compression_ratio": len(tok.encode(val_text)) / len(val_text),
            "completions": completions,
        }

    print(f"{'Metric':30}{'raw':>15}{'glyph':>15}")
    print(f"{'Perplexity (val)':30}{results['raw']['perplexity']:>15.3f}{results['glyph']['perplexity']:>15.3f}")
    print(f"{'Tokens/sec (inference)':30}{results['raw']['tokens_per_second']:>15.2f}{results['glyph']['tokens_per_second']:>15.2f}")
    print(f"{'Avg tokens/line (val)':30}{results['raw']['avg_tokens_per_line']:>15.2f}{results['glyph']['avg_tokens_per_line']:>15.2f}")
    print(f"{'Compression ratio (val)':30}{results['raw']['compression_ratio']:>15.4f}{results['glyph']['compression_ratio']:>15.4f}")

    for label, _, _ in MODELS:
        print(f"\n--- {label} sample completions ---")
        for c in results[label]["completions"]:
            print(c)
            print("---")


def _avg_tokens_per_line(tok, text: str) -> float:
    lines = [line for line in text.split("\n") if line.strip()]
    total = sum(len(tok.encode(line)) for line in lines)
    return total / len(lines)


if __name__ == "__main__":
    main()
