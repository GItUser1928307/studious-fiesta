#!/usr/bin/env python3
"""Train the model on the conversational dataset."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import create_model
from tokenizer import CharTokenizer
from config import ModelConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")
CKPT_DIR = os.path.join(PARENT_DIR, "quick_ckpt")

FAST_CONFIG = ModelConfig(
    vocab_size=128,
    hidden_size=128,
    num_layers=4,
    num_heads=4,
    intermediate_size=256,
    max_seq_len=64,
)

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

def main():
    device = "cpu"
    model_config = FAST_CONFIG
    tokenizer = CharTokenizer()
    model = create_model(model_config).to(device)
    print(f"Params: {model.count_params():,}", flush=True)

    dataset = TextDataset(DATA_FILE, tokenizer, model_config.max_seq_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    os.makedirs(CKPT_DIR, exist_ok=True)
    model.train()

    max_steps = 200
    start = time.time()
    for step in range(max_steps):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % 20 == 0:
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | {time.time()-start:.1f}s", flush=True)

    save_path = os.path.join(CKPT_DIR, "best.pt")
    torch.save({"model": model.state_dict(), "config": model_config}, save_path)
    print(f"\nDone! {time.time()-start:.1f}s | Final loss: {loss.item():.4f}", flush=True)
    print(f"Saved to {save_path}", flush=True)

if __name__ == "__main__":
    main()
