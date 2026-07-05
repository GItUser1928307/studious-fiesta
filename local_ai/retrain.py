#!/usr/bin/env python3
"""Train the model — auto-launches DDP when multiple GPUs detected."""
import sys, os, time, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import create_model
from tokenizer import get_tokenizer, save_tokenizer
from config import auto_config_from_data, auto_train_config

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
        x = torch.zeros(self.seq_len, dtype=torch.long)
        y = torch.zeros(self.seq_len, dtype=torch.long)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        n = len(full) - 1
        x[:n] = torch.tensor(full[:-1], dtype=torch.long)
        y[:n] = torch.tensor(full[1:], dtype=torch.long)
        mask[len(input_ids) - 1: n] = True
        return x, y, mask


def auto_launch():
    """If called via plain `python retrain.py` and 2+ GPUs exist, re-launch with torchrun."""
    if "RANK" in os.environ:
        return
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return
    print(f"Detected {num_gpus} GPUs — launching with torchrun for DDP...", flush=True)
    script = os.path.abspath(__file__)
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={num_gpus}", script] + sys.argv[1:]
    proc = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(num_gpus))})
    sys.exit(proc.returncode)


def main():
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    is_main = (rank == 0)

    if is_main:
        num_gpus = torch.cuda.device_count()
        print(f"GPUs detected: {num_gpus}", flush=True)
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
        if use_ddp:
            print(f"Using DDP across {world_size} GPUs — each GPU runs its own process", flush=True)
        elif num_gpus == 1:
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("No GPU detected, using CPU", flush=True)

    tokenizer_name = "word"
    tokenizer = get_tokenizer(tokenizer_name, data_file=DATA_FILE)
    if is_main:
        print(f"Tokenizer: {tokenizer_name}", flush=True)
        print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
        print(tokenizer.vocab_info(), flush=True)

    model_config = auto_config_from_data(DATA_FILE)
    train_config = auto_train_config(DATA_FILE, CKPT_DIR)
    model = create_model(model_config).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    params = model.module.count_params() if hasattr(model, 'module') else model.count_params()
    if is_main:
        print(f"Params: {params:,}", flush=True)
        print(f"Config: hidden={model_config.hidden_size}, layers={model_config.num_layers}, seq={model_config.max_seq_len}, vocab={model_config.vocab_size}", flush=True)
        print(f"Device: {device}", flush=True)

    dataset = TextDataset(DATA_FILE, tokenizer, model_config.max_seq_len)
    if len(dataset) == 0:
        if is_main:
            print("ERROR: No valid <q>/<a> pairs found in dataset!", flush=True)
        if use_ddp:
            dist.destroy_process_group()
        return

    batch_size = min(train_config.batch_size, len(dataset))
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=(os.cpu_count() or 1) > 1,
    )
    if is_main:
        print(f"Dataset: {len(dataset)} samples, {len(loader)} batches", flush=True)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=0.1)
    warmup_steps = min(100, train_config.max_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, train_config.max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(CKPT_DIR, exist_ok=True)
    model.train()

    max_steps = train_config.max_steps
    start_step = 0
    start = time.time()

    resume_path = os.path.join(CKPT_DIR, "latest.pt")
    if "--resume" in sys.argv and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        if is_main:
            print(f"Resumed from step {start_step}", flush=True)

    step = start_step
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for x, y, mask in loader:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, loss = model(x, y, loss_mask=mask)
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if is_main and step % train_config.log_interval == 0:
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | {time.time()-start:.1f}s", flush=True)
            if is_main and step % train_config.save_interval == 0:
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    "model": state_dict,
                    "config": model_config,
                    "tokenizer": tokenizer_name,
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }, os.path.join(CKPT_DIR, "latest.pt"))
                print(f"  Saved checkpoint at step {step}", flush=True)
            if step >= max_steps:
                break

    if is_main:
        save_path = os.path.join(CKPT_DIR, "best.pt")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({"model": state_dict, "config": model_config, "tokenizer": tokenizer_name}, save_path)
        save_tokenizer(tokenizer, os.path.join(CKPT_DIR, "tokenizer.json"))
        print(f"\nDone! {time.time()-start:.1f}s | Final loss: {loss.item():.4f}", flush=True)
        print(f"Saved to {save_path}", flush=True)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    auto_launch()
    main()
