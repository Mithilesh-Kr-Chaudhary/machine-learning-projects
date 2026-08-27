"""Generate text from a trained Task 5 character-level RNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from train_rnn import CharRNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--seed_text", default="THE ")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--output", default="generated_text.txt")
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero.")

    model_dir = Path(args.model_dir)
    vocab = json.loads((model_dir / "vocabulary.json").read_text(encoding="utf-8"))
    char_to_id = {char: idx for idx, char in enumerate(vocab)}
    unknown = sorted(set(args.seed_text) - set(vocab))
    if unknown:
        raise ValueError(f"Seed contains characters absent from the corpus: {unknown}")
    checkpoint = torch.load(model_dir / "char_rnn.pt", map_location="cpu", weights_only=True)
    model = CharRNN(checkpoint["vocab_size"], checkpoint["embedding_dim"],
                    checkpoint["hidden_size"], checkpoint["layers"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    generated = args.seed_text
    state = None
    with torch.no_grad():
        token = torch.tensor([[char_to_id[c] for c in args.seed_text]], dtype=torch.long)
        logits, state = model(token, state)
        next_logits = logits[:, -1, :]
        for _ in range(args.length):
            probabilities = F.softmax(next_logits / args.temperature, dim=-1)
            token = torch.multinomial(probabilities, 1)
            generated += vocab[token.item()]
            logits, state = model(token, state)
            next_logits = logits[:, -1, :]

    Path(args.output).write_text(generated, encoding="utf-8")
    print(generated)
    print(f"\nSaved generated text to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
