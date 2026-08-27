"""Train a character-level LSTM language model for Task 5.

Example:
    python train_rnn.py --text_file corpus.txt --epochs 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CharacterDataset(Dataset):
    def __init__(self, encoded: list[int], sequence_length: int):
        if len(encoded) <= sequence_length:
            raise ValueError("The text corpus must be longer than --sequence_length.")
        self.encoded = torch.tensor(encoded, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.encoded) - self.sequence_length

    def __getitem__(self, index: int):
        x = self.encoded[index : index + self.sequence_length]
        y = self.encoded[index + 1 : index + self.sequence_length + 1]
        return x, y


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, layers, batch_first=True,
                            dropout=0.2 if layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, state=None):
        x = self.embedding(tokens)
        x, state = self.lstm(x, state)
        return self.output(x), state


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", default="corpus.txt")
    parser.add_argument("--output_dir", default="model")
    parser.add_argument("--sequence_length", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--embedding_dim", type=int, default=96)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    text = Path(args.text_file).read_text(encoding="utf-8").replace("\r\n", "\n")
    chars = sorted(set(text))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    encoded = [char_to_id[char] for char in text]
    dataset = CharacterDataset(encoded, args.sequence_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharRNN(len(chars), args.embedding_dim, args.hidden_size, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, len(chars)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}/{args.epochs} | loss: {total_loss / len(loader):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "vocab_size": len(chars),
                "embedding_dim": args.embedding_dim, "hidden_size": args.hidden_size,
                "layers": args.layers}, output_dir / "char_rnn.pt")
    (output_dir / "vocabulary.json").write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")
    print(f"Saved model and vocabulary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
