#!/usr/bin/env python3
"""Train the model - auto-detects hardware."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import create_model
from tokenizer import WordTokenizer
from config import auto_config_from_data, auto_train_config
from system import print_system_info, get_cpu_threads

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")
CKPT_DIR = os.path.join(PARENT_DIR, "quick_ckpt")


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.tokens = []
        for line in lines:
            line = line.strip()
            if line:
                self.tokens.extend(tokenizer.encode(line))
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def main():
    info = print_system_info()
    torch.set_num_threads(info["threads"])

    tokenizer = WordTokenizer.build(DATA_FILE)
    print(f"Tokenizer: {tokenizer.vocab_size} words", flush=True)
    print(tokenizer.vocab_info(), flush=True)

    model_config = auto_config_from_data(DATA_FILE)
    train_config = auto_train_config(DATA_FILE, CKPT_DIR)
    model = create_model(model_config)
    print(f"Params: {model.count_params():,}", flush=True)
    print(f"Config: hidden={model_config.hidden_size}, layers={model_config.num_layers}, seq={model_config.max_seq_len}, vocab={model_config.vocab_size}", flush=True)

    dataset = TextDataset(DATA_FILE, tokenizer, model_config.max_seq_len)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=0.1)
    os.makedirs(CKPT_DIR, exist_ok=True)
    model.train()

    max_steps = train_config.max_steps
    start = time.time()
    step = 0
    while step < max_steps:
        for x, y in loader:
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % train_config.log_interval == 0:
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | {time.time()-start:.1f}s", flush=True)
            if step >= max_steps:
                break

    save_path = os.path.join(CKPT_DIR, "best.pt")
    torch.save({"model": model.state_dict(), "config": model_config}, save_path)
    tokenizer.save(os.path.join(CKPT_DIR, "tokenizer.json"))
    print(f"\nDone! {time.time()-start:.1f}s | Final loss: {loss.item():.4f}", flush=True)
    print(f"Saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
