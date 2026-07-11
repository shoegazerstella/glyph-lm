"""Interactive REPL for raw model (no encoding/decoding).

Usage: python -m glyph.chat_raw
"""

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from glyph.train import get_device

MODEL_DIR = "models/model_raw"


def main() -> None:
    device = get_device()
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{MODEL_DIR}/tokenizer.json", pad_token="<pad>")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("Raw model chat — type in plain English, Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue

        input_ids = torch.tensor([tok.encode(user_input)]).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=60, do_sample=True, top_k=50, temperature=0.8
            )
        output = tok.decode(out[0].tolist())
        print(f"raw> {output}\n")


if __name__ == "__main__":
    main()
