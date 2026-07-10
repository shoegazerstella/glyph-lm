"""Tiny GPT-2 training on raw vs glyph corpora (see docs/prd.md)."""

import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

BLOCK_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4
LOG_EVERY = 100


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BlockDataset(Dataset):
    # GLYPH NOTE: concatenate the whole corpus into one token stream and chunk into
    # fixed-size blocks, rather than padding each line to block_size. The glyph
    # corpus naturally yields fewer tokens per line than raw text, so per-line
    # padding would waste more compute on the raw side and bias the training-time
    # comparison. Concatenate-and-chunk keeps token throughput comparable.
    def __init__(self, ids: list[int], block_size: int = BLOCK_SIZE):
        self.examples = [
            ids[i : i + block_size] for i in range(0, len(ids) - block_size, block_size)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.examples[idx], dtype=torch.long)
        return x, x.clone()


def train_model(corpus_path: str, tokenizer_dir: str, out_dir: str, label: str, seed: int = 42) -> float:
    set_seed(seed)
    device = get_device()
    tok = PreTrainedTokenizerFast(
        tokenizer_file=f"{tokenizer_dir}/tokenizer.json", pad_token="<pad>"
    )

    with open(corpus_path) as f:
        ids = tok.encode(f.read())

    dataset = BlockDataset(ids)
    # Explicit generator for reproducible shuffling across platforms
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    # GLYPH NOTE: use the tokenizer's actual trained vocab size (may be < 2048)
    # rather than hardcoding it, to avoid an embedding/vocab size mismatch.
    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=BLOCK_SIZE,
        n_embd=128,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    step = 0
    start = time.time()
    loss = torch.tensor(0.0)
    for epoch in range(EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(input_ids=x, labels=y)
            loss = out.loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % LOG_EVERY == 0:
                elapsed = time.time() - start
                print(
                    f"[{label}] epoch={epoch} step={step} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
                )
            step += 1

    final_loss = loss.item()
    total_time = time.time() - start
    print(f"[{label}] final_loss={final_loss:.4f} total_time={total_time:.1f}s")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(f"{out_dir}/train_metrics.json", "w") as f:
        json.dump({"final_loss": final_loss, "train_time_seconds": total_time, "seed": seed}, f, indent=2)
    return final_loss


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"Training with seed={seed}")
    train_model("data/raw/train.txt", "tokenizers/raw_bpe", "models/model_raw", "raw", seed=seed)
    train_model("data/glyph/train.txt", "tokenizers/glyph_bpe", "models/model_glyph", "glyph", seed=seed)
