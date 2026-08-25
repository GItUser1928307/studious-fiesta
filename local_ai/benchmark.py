#!/usr/bin/env python3
"""
Lightweight benchmark for 2× Tesla T4 (Kaggle).

Measures steady-state throughput after warmup, excluding
checkpoint loading and torch.compile warmup.

Run on Kaggle (2× T4):
    !torchrun --standalone --nnodes=1 --nproc-per-node=2 benchmark.py
    !torchrun --standalone --nnodes=1 --nproc-per-node=2 benchmark.py --compile
    !torchrun --standalone --nnodes=1 --nproc-per-node=2 benchmark.py --batch-size 64

Compares:
    - steps/sec
    - tokens/sec
    - samples/sec
    - GPU memory (via torch.cuda.max_memory_allocated)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import create_model
from tokenizer import get_tokenizer
from config import auto_config_from_data, auto_train_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")

# Reuse dataset from retrain.py
from retrain import TextDataset, build_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--compile", action="store_true", help="enable torch.compile")
    p.add_argument("--no-amp", action="store_true", help="disable AMP")
    p.add_argument("--steps", type=int, default=100, help="benchmark steps (steady-state)")
    p.add_argument("--warmup", type=int, default=10, help="warmup steps excluded from measurement")
    return p.parse_args()


def main():
    args = parse_args()

    # DDP init
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model_config = auto_config_from_data(DATA_FILE)
    train_config = auto_train_config(DATA_FILE, os.path.join(PARENT_DIR, "quick_ckpt"))

    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.workers is not None:
        train_config.num_workers = args.workers
    if args.compile:
        train_config.compile_enabled = True
    if args.no_amp:
        train_config.amp_enabled = False

    tokenizer = get_tokenizer("word", data_file=DATA_FILE)
    dataset = TextDataset(DATA_FILE, tokenizer, model_config.max_seq_len)

    model = create_model(model_config).to(device)
    if train_config.compile_enabled and device.type == "cuda":
        try:
            model = torch.compile(model, mode=train_config.compile_mode)
            if is_main:
                print(f"[benchmark] torch.compile enabled ({train_config.compile_mode})")
        except Exception as e:
            if is_main:
                print(f"[benchmark] compile failed: {e}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    gradient_as_bucket_view=True, static_graph=True)

    batch_size = min(train_config.batch_size, len(dataset))
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    num_workers = min(getattr(train_config, "num_workers", 2), os.cpu_count() or 1)
    if use_ddp and world_size > 1:
        num_workers = min(num_workers, 2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=getattr(train_config, "prefetch_factor", 2),
        drop_last=True,
    )

    use_amp = getattr(train_config, "amp_enabled", True) and device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = build_scheduler(optimizer, train_config.max_steps)

    if is_main:
        print(f"[benchmark] model {model_config.hidden_size}h {model_config.num_layers}L "
              f"{model_config.num_heads}H seq={model_config.max_seq_len} vocab={model_config.vocab_size}")
        print(f"[benchmark] batch {batch_size} per GPU (effective {batch_size*world_size}) "
              f"workers={num_workers} amp={'fp16' if use_amp else 'fp32'} compile={train_config.compile_enabled}")
        print(f"[benchmark] warmup {args.warmup} steps, measured {args.steps} steps")

    model.train()
    total_steps = args.warmup + args.steps
    step = 0
    epoch = 0

    # Warmup + benchmark
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.time()
    measured_start = None
    measured_steps = 0

    while step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, y, mask in loader:
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")
            mask = mask.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y, loss_mask=mask)
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            if step == args.warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                measured_start = time.time()
                measured_steps = 0
                if is_main:
                    print(f"[benchmark] warmup done, measuring {args.steps} steps...")
            if step > args.warmup:
                measured_steps += 1

            if step >= total_steps:
                break
        epoch += 1
        if step >= total_steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - measured_start if measured_start else time.time() - t0

    it_per_sec = measured_steps / max(0.001, elapsed)
    tokens_per_sec = it_per_sec * batch_size * model_config.max_seq_len * world_size
    samples_per_sec = it_per_sec * batch_size * world_size

    if is_main:
        print(f"\n[benchmark] RESULT ({'compile' if train_config.compile_enabled else 'no-compile'}):")
        print(f"  Steps/sec:   {it_per_sec:.2f}")
        print(f"  Tokens/sec:  {tokens_per_sec:.0f}")
        print(f"  Samples/sec: {samples_per_sec:.0f}")
        print(f"  Batch/GPU:   {batch_size}  Effective: {batch_size*world_size}")
        print(f"  Seq len:     {model_config.max_seq_len}")
        print(f"  Elapsed:     {elapsed:.2f}s for {measured_steps} steps")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  Peak VRAM:   {peak:.2f} GB per GPU")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i} {torch.cuda.get_device_name(i)}: "
                      f"{torch.cuda.max_memory_allocated(i)/1024**3:.2f} GB peak")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
