"""Compare the raw and glyph models on perplexity, speed, and sample output."""

import json
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


def compute_perplexity(model, tok, text: str, device, block_size: int = BLOCK_SIZE) -> tuple[float, float]:
    """Returns (perplexity, bits_per_char).

    bits_per_char = total cross-entropy loss in bits / original character count.
    This normalizes for different tokenizations — shorthand tokens carry more info
    per token, so per-token perplexity is misleading across raw vs glyph models.
    """
    ids = tok.encode(text)
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(ids) - block_size, block_size):
            chunk = torch.tensor(ids[i : i + block_size]).unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk)
            losses.append(out.loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)

    # Convert nats to bits, normalize by original char count
    total_loss_bits = mean_loss * len(losses) * block_size / math.log(2)
    bits_per_char = total_loss_bits / len(text)

    return perplexity, bits_per_char


def inference_speed(model, tok, prompt: str, device, max_new_tokens: int = 100) -> tuple[float, float, float]:
    """Returns (tokens_per_sec, chars_per_sec, e2e_chars_per_sec).

    - tokens_per_sec: BPE tokens generated per second (model only)
    - chars_per_sec: character throughput after BPE decode (model only)
    - e2e_chars_per_sec: end-to-end character throughput including steno encode/decode
      (only applicable to glyph model; equals chars_per_sec for raw model)
    """
    input_ids = torch.tensor([tok.encode(prompt)]).to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - start

    n_generated = out.shape[1] - input_ids.shape[1]
    tokens_per_sec = n_generated / elapsed

    # Decode generated portion to measure character throughput
    generated_ids = out[0, input_ids.shape[1]:].tolist()
    generated_text = tok.decode(generated_ids)
    chars_per_sec = len(generated_text) / elapsed

    return tokens_per_sec, chars_per_sec, chars_per_sec  # e2e same as chars_per_sec (no overhead measured here)


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

        train_metrics_path = f"{model_dir}/train_metrics.json"
        try:
            with open(train_metrics_path) as f:
                train_metrics = json.load(f)
        except FileNotFoundError:
            train_metrics = {"final_loss": None, "train_time_seconds": None}

        perplexity, bits_per_char = compute_perplexity(model, tok, val_text, device)
        tokens_per_sec, chars_per_sec, _ = inference_speed(model, tok, prompt, device)

        # End-to-end timing: include steno encode/decode overhead for glyph
        e2e_chars_per_sec = chars_per_sec  # default: no overhead
        if label == "glyph":
            plain_prompt = SEED_PROMPT
            start_e2e = time.time()
            # Encode input
            encoded_prompt = encode(plain_prompt)
            input_ids = torch.tensor([tok.encode(encoded_prompt)]).to(device)
            # Generate
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=100, do_sample=False)
            # Decode output
            generated_text = tok.decode(out[0].tolist())
            decoded_text = decode(generated_text)
            elapsed_e2e = time.time() - start_e2e
            e2e_chars_per_sec = len(decoded_text) / elapsed_e2e

        results[label] = {
            "final_train_loss": train_metrics["final_loss"],
            "train_time_seconds": train_metrics["train_time_seconds"],
            "perplexity": perplexity,
            "bits_per_char": bits_per_char,
            "tokens_per_second": tokens_per_sec,
            "chars_per_second": chars_per_sec,
            "e2e_chars_per_second": e2e_chars_per_sec,
            "avg_tokens_per_line": _avg_tokens_per_line(tok, val_text),
            "compression_ratio": len(tok.encode(val_text)) / len(val_text),
            "completions": completions,
        }

    def _row(label_text: str, key: str, fmt: str = "{:>15.3f}") -> str:
        raw_val, glyph_val = results["raw"][key], results["glyph"][key]
        raw_str = fmt.format(raw_val) if raw_val is not None else f"{'n/a':>15}"
        glyph_str = fmt.format(glyph_val) if glyph_val is not None else f"{'n/a':>15}"
        return f"{label_text:30}{raw_str}{glyph_str}"

    print(f"{'Metric':30}{'raw':>15}{'glyph':>15}")
    print(_row("Final train loss", "final_train_loss"))
    print(_row("Perplexity (val)", "perplexity"))
    print(_row("Bits/char (val)", "bits_per_char"))
    print(_row("Tokens/sec (inference)", "tokens_per_second", "{:>15.2f}"))
    print(_row("Chars/sec (inference)", "chars_per_second", "{:>15.2f}"))
    print(_row("E2E chars/sec", "e2e_chars_per_second", "{:>15.2f}"))
    print(_row("Avg tokens/line (val)", "avg_tokens_per_line", "{:>15.2f}"))
    print(_row("Compression ratio (val)", "compression_ratio", "{:>15.4f}"))

    for label, _, _ in MODELS:
        print(f"\n--- {label} sample completions ---")
        for c in results[label]["completions"]:
            print(c)
            print("---")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json")


def _avg_tokens_per_line(tok, text: str) -> float:
    lines = [line for line in text.split("\n") if line.strip()]
    total = sum(len(tok.encode(line)) for line in lines)
    return total / len(lines)


if __name__ == "__main__":
    main()
