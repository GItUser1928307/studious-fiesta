import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import create_model
from tokenizer import GPT2Tokenizer, CharTokenizer
from config import ModelConfig, TrainConfig, SMALL_CONFIG


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, seq_len: int, max_examples: int = None):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
        if max_examples:
            max_tokens = max_examples * seq_len
            self.tokens = self.tokens[:max_tokens]

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_lr(it: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if it < warmup:
        return max_lr * (it + 1) / (warmup + 1)
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(model: nn.Module, tokenizer, config: TrainConfig, model_config: ModelConfig, device: str):
    dataset = TextDataset(config.data_file, tokenizer, model_config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    os.makedirs(config.save_dir, exist_ok=True)

    model.train()
    total_steps = 0
    best_loss = float("inf")
    start_time = time.time()

    while total_steps < config.max_steps:
        for x, y in loader:
            if total_steps >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)
            lr = get_lr(total_steps, config.warmup_steps, config.max_steps, config.learning_rate, config.learning_rate * 0.1)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            logits, loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optim.step()

            if total_steps % config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (total_steps * config.batch_size * model_config.max_seq_len) / max(elapsed, 1)
                print(f"step {total_steps}/{config.max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tokens_per_sec:.0f}")

            if total_steps % config.save_interval == 0 and total_steps > 0:
                path = os.path.join(config.save_dir, f"step_{total_steps}.pt")
                torch.save({"model": model.state_dict(), "step": total_steps, "loss": loss.item()}, path)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = os.path.join(config.save_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "step": total_steps, "loss": loss.item()}, best_path)

            total_steps += 1

    end_time = time.time()
    print(f"\nTraining complete! Time: {end_time - start_time:.1f}s")
    final_path = os.path.join(config.save_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": model_config}, final_path)
    print(f"Model saved to {final_path}")


def train_quick(device: str):
    print("=== Quick Demo Training (Character-Level) ===")
    model_config = SMALL_CONFIG
    tokenizer = CharTokenizer()
    model = create_model(model_config).to(device)
    print(f"Model params: {model.count_params():,}")

    sample_text = """Once upon a time there was a little girl named Lily. She lived in a small house near a big forest. Every day she would play in the garden with her cat named Whiskers. Whiskers was a fluffy orange cat with big green eyes.

One sunny morning, Lily decided to explore the forest. She packed some cookies and juice in a small basket. Whiskers followed her as she walked into the trees.

Deep in the forest they found a sparkling stream. Fish of many colors swam in the clear water. Lily sat on a mossy rock and shared her cookies with Whiskers.

Suddenly they heard a strange noise. It came from behind a large bush. Lily was scared but curious. She slowly walked toward the bush and peeked through the leaves.

To her surprise she saw a tiny baby deer stuck in some vines. Its mother was nowhere to be seen. Lily carefully untangled the vines and freed the deer.

The baby deer licked her hand and ran off into the forest. Lily felt happy that she could help. She and Whiskers returned home just before sunset.

From that day on Lily visited the forest every weekend. She made many animal friends and learned that kindness makes the world a better place."""

    data_path = "quick_train_data.txt"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    train_config = TrainConfig(
        data_file=data_path,
        batch_size=2,
        learning_rate=3e-4,
        max_steps=500,
        log_interval=25,
        save_interval=200,
        save_dir="quick_ckpt",
    )
    train(model, tokenizer, train_config, model_config, device)

    return model, tokenizer


def train_full(device: str):
    print("=== Full Training (BPE Tokenizer) ===")
    model_config = ModelConfig()
    tokenizer = GPT2Tokenizer()
    model = create_model(model_config).to(device)
    print(f"Model params: {model.count_params():,}")
    print(f"Architecture: {model_config.num_layers} layers, {model_config.hidden_size} hidden, {model_config.num_heads} heads")
    print(f"Model size: ~{model_config.total_params / 1e6:.1f}M parameters")

    train_config = TrainConfig()
    train(model, tokenizer, train_config, model_config, device)

    return model, tokenizer
