#!/usr/bin/env python3
"""Train the model - auto-detects hardware."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import create_model
from tokenizer import get_tokenizer, save_tokenizer, list_tokenizers
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

        self.samples = []
        self.seq_len = seq_len
        i = 0
        while i < len(lines) - 1:
            q_line = lines[i].strip()
            a_line = lines[i + 1].strip()
            if q_line.startswith("<q>") and a_line.startswith("<a>"):
                q_tokens = [t for t in tokenizer.tokenize(q_line) if t not in ("<q>", "<a>")]
                a_tokens = [t for t in tokenizer.tokenize(a_line) if t not in ("<q>", "<a>")]

                input_ids = [tokenizer.bos_token_id, tokenizer.q_token_id]
                input_ids += [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in q_tokens]
                input_ids.append(tokenizer.a_token_id)

                target_ids = [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in a_tokens]
                target_ids.append(tokenizer.eos_token_id)

                full = input_ids + target_ids
                if len(full) <= seq_len + 1:
                    self.samples.append((input_ids, target_ids))
                i += 2
            else:
                i += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, target_ids = self.samples[idx]
        full = input_ids + target_ids
        full = full[:self.seq_len + 1]
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)
        mask = torch.zeros(len(full) - 1, dtype=torch.bool)
        mask[len(input_ids) - 1:] = True
        return x, y, mask


def main():
    info = print_system_info()
    torch.set_num_threads(info["threads"])

    tokenizer_name = "word"
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--tokenizer" and i + 1 < len(args):
            tokenizer_name = args[i + 1]

    print(f"Tokenizer: {tokenizer_name}", flush=True)
    tokenizer = get_tokenizer(tokenizer_name, data_file=DATA_FILE)
    print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
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
        for x, y, mask in loader:
            logits, loss = model(x, y, loss_mask=mask)
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
    torch.save({"model": model.state_dict(), "config": model_config, "tokenizer": tokenizer_name}, save_path)
    save_tokenizer(tokenizer, os.path.join(CKPT_DIR, "tokenizer.json"))
    print(f"\nDone! {time.time()-start:.1f}s | Final loss: {loss.item():.4f}", flush=True)
    print(f"Saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
