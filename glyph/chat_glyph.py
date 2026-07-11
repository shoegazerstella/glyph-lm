"""Interactive REPL for glyph model.

Your input is shorthand-encoded before being fed to the model (since that's the
only "language" model_glyph was trained on), and the model's shorthand output is
decoded back to natural language before being printed.

Usage: python -m glyph.chat_glyph
"""

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from glyph.encoder import decode, encode
from glyph.train import get_device

MODEL_DIR = "models/model_glyph"


def main() -> None:
    device = get_device()
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{MODEL_DIR}/tokenizer.json", pad_token="<pad>")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("GlyphLM chat — type in plain English, Ctrl+C to quit.")
    print("(Reminder: this is a ~4M-param model trained on 1MB of Shakespeare — expect noise, not coherence.)\n")

    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue

        glyph_prompt = encode(user_input)
        input_ids = torch.tensor([tok.encode(glyph_prompt)]).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=60, do_sample=True, top_k=50, temperature=0.8
            )
        glyph_output = tok.decode(out[0].tolist())
        print(f"glyph> {glyph_output}")
        print(f"plain> {decode(glyph_output)}\n")


if __name__ == "__main__":
    main()
