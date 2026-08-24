#!/usr/bin/env python3
"""
Train the model on CPU, NVIDIA CUDA GPUs, or TPU v5e-8 via PyTorch/XLA.

GPU:
    python retrain.py

TPU:
    python retrain.py --tpu

Resume:
    python retrain.py --resume
    python retrain.py --tpu --resume
"""

import sys
import os
import time
import math
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import create_model
from tokenizer import get_tokenizer, save_tokenizer
from config import auto_config_from_data, auto_train_config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")
CKPT_DIR = os.path.join(PARENT_DIR, "quick_ckpt")


# ============================================================
# Dataset
# ============================================================

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
                q_tokens = [
                    t
                    for t in tokenizer.tokenize(q_line)
                    if t not in ("<q>", "<a>")
                ]

                a_tokens = [
                    t
                    for t in tokenizer.tokenize(a_line)
                    if t not in ("<q>", "<a>")
                ]

                input_ids = [
                    tokenizer.bos_token_id,
                    tokenizer.q_token_id,
                ]

                input_ids += [
                    tokenizer.word_to_id.get(
                        t,
                        tokenizer.unk_token_id
                    )
                    for t in q_tokens
                ]

                input_ids.append(tokenizer.a_token_id)

                target_ids = [
                    tokenizer.word_to_id.get(
                        t,
                        tokenizer.unk_token_id
                    )
                    for t in a_tokens
                ]

                target_ids.append(tokenizer.eos_token_id)

                full = input_ids + target_ids

                if len(full) <= seq_len + 1:
                    x = torch.zeros(seq_len, dtype=torch.long)
                    y = torch.zeros(seq_len, dtype=torch.long)
                    mask = torch.zeros(seq_len, dtype=torch.bool)

                    n = len(full) - 1

                    x[:n] = torch.tensor(
                        full[:-1],
                        dtype=torch.long
                    )

                    y[:n] = torch.tensor(
                        full[1:],
                        dtype=torch.long
                    )

                    mask[len(input_ids) - 1:n] = True

                    self.samples.append((x, y, mask))

                i += 2

            else:
                i += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Shared utilities
# ============================================================

def build_scheduler(optimizer, max_steps):
    warmup_steps = min(100, max_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (
            (step - warmup_steps)
            / max(1, max_steps - warmup_steps)
        )

        return (
            0.1
            + 0.9
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )


def create_dataset_and_tokenizer(model_config):
    tokenizer_name = "word"

    tokenizer = get_tokenizer(
        tokenizer_name,
        data_file=DATA_FILE
    )

    dataset = TextDataset(
        DATA_FILE,
        tokenizer,
        model_config.max_seq_len
    )

    return tokenizer_name, tokenizer, dataset


def print_model_information(
    model,
    model_config,
    tokenizer,
    device,
    rank=0,
    world_size=1,
    accelerator_name=None,
):
    if rank != 0:
        return

    params = model.count_params()

    print(
        f"Params: {params:,}",
        flush=True
    )

    print(
        f"Config: "
        f"hidden={model_config.hidden_size}, "
        f"layers={model_config.num_layers}, "
        f"seq={model_config.max_seq_len}, "
        f"vocab={model_config.vocab_size}",
        flush=True
    )

    print(
        f"Device: {device}",
        flush=True
    )

    if accelerator_name:
        print(
            f"Accelerator: {accelerator_name}",
            flush=True
        )

    print(
        f"World size: {world_size}",
        flush=True
    )

    print(
        f"Tokenizer: word",
        flush=True
    )

    print(
        f"Vocab size: {tokenizer.vocab_size}",
        flush=True
    )

    print(
        tokenizer.vocab_info(),
        flush=True
    )


# ============================================================
# CUDA / GPU training
# ============================================================

def auto_launch_gpu():
    """
    Preserve the original CUDA behavior.

    If plain:
        python retrain.py

    detects multiple GPUs, relaunch using torch.distributed.run.
    """

    if "--tpu" in sys.argv:
        return

    if "RANK" in os.environ:
        return

    num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        return

    print(
        f"Detected {num_gpus} GPUs — "
        f"launching with torchrun for DDP...",
        flush=True
    )

    script = os.path.abspath(__file__)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        script,
    ]

    cmd += [
        arg
        for arg in sys.argv[1:]
        if arg != "--tpu"
    ]

    proc = subprocess.run(
        cmd,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(i) for i in range(num_gpus)
            ),
        },
    )

    sys.exit(proc.returncode)


def train_gpu():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "0")

    torch.backends.cudnn.benchmark = True

    use_ddp = "RANK" in os.environ

    if use_ddp:
        dist.init_process_group("nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        local_rank = int(
            os.environ["LOCAL_RANK"]
        )

        device = torch.device(
            f"cuda:{local_rank}"
        )

        torch.cuda.set_device(device)

    else:
        rank = 0
        world_size = 1
        local_rank = 0

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    is_main = rank == 0

    if is_main:
        num_gpus = torch.cuda.device_count()

        print(
            f"GPUs detected: {num_gpus}",
            flush=True
        )

        for i in range(num_gpus):
            print(
                f"  GPU {i}: "
                f"{torch.cuda.get_device_name(i)}",
                flush=True
            )

        if use_ddp:
            print(
                f"Using DDP across {world_size} GPUs",
                flush=True
            )

        elif num_gpus == 1:
            print(
                f"Using single GPU: "
                f"{torch.cuda.get_device_name(0)}",
                flush=True
            )

        else:
            print(
                "No GPU detected, using CPU",
                flush=True
            )

    tokenizer_name, tokenizer, dataset = (
        create_dataset_and_tokenizer(
            auto_config_from_data(DATA_FILE)
        )
    )

    if is_main:
        print(
            f"Tokenizer: {tokenizer_name}",
            flush=True
        )

        print(
            f"Vocab size: {tokenizer.vocab_size}",
            flush=True
        )

        print(
            tokenizer.vocab_info(),
            flush=True
        )

    model_config = auto_config_from_data(DATA_FILE)
    train_config = auto_train_config(
        DATA_FILE,
        CKPT_DIR
    )

    if len(dataset) == 0:
        if is_main:
            print(
                "ERROR: No valid <q>/<a> pairs found "
                "in dataset!",
                flush=True
            )

        if use_ddp:
            dist.destroy_process_group()

        return

    model = create_model(
        model_config
    ).to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    model_for_count = (
        model.module
        if hasattr(model, "module")
        else model
    )

    print_model_information(
        model_for_count,
        model_config,
        tokenizer,
        device,
        rank,
        world_size,
        "NVIDIA CUDA",
    )

    batch_size = min(
        train_config.batch_size,
        len(dataset)
    )

    sampler = (
        DistributedSampler(
            dataset,
            shuffle=True
        )
        if use_ddp
        else None
    )

    num_workers = min(
        8,
        os.cpu_count() or 1
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(
        dataset,
        **loader_kwargs
    )

    if is_main:
        print(
            f"Dataset: {len(dataset)} samples, "
            f"{len(loader)} batches",
            flush=True
        )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        train_config.max_steps
    )

    os.makedirs(
        CKPT_DIR,
        exist_ok=True
    )

    model.train()

    max_steps = train_config.max_steps
    start_step = 0

    start = time.time()

    resume_path = os.path.join(
        CKPT_DIR,
        "latest.pt"
    )

    if (
        "--resume" in sys.argv
        and os.path.exists(resume_path)
    ):
        ckpt = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        model_to_load = (
            model.module
            if hasattr(model, "module")
            else model
        )

        model_to_load.load_state_dict(
            ckpt["model"]
        )

        start_step = ckpt["step"]

        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler"]
        )

        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(
                ckpt["scaler"]
            )

        if is_main:
            print(
                f"Resumed from step {start_step}",
                flush=True
            )

    step = start_step

    loss_sum = 0.0
    loss_count = 0

    step_start = time.time()

    while step < max_steps:

        if sampler is not None:
            sampler.set_epoch(step)

        for x, y, mask in loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            mask = mask.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp
            ):
                logits, loss = model(
                    x,
                    y,
                    loss_mask=mask
                )

                loss = loss.mean()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.gradient_clip
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            step += 1

            loss_sum += loss.item()
            loss_count += 1

            if (
                is_main
                and step % train_config.log_interval == 0
            ):
                elapsed = time.time() - start

                it_per_sec = (
                    loss_count
                    / max(
                        0.001,
                        time.time() - step_start
                    )
                )

                avg_loss = (
                    loss_sum
                    / max(1, loss_count)
                )

                pct = (
                    step
                    / max_steps
                    * 100
                )

                remaining = (
                    max_steps - step
                ) / max(
                    0.001,
                    it_per_sec
                )

                m, s = divmod(
                    int(elapsed),
                    60
                )

                h, m = divmod(
                    m,
                    60
                )

                rm, rs = divmod(
                    int(remaining),
                    60
                )

                rh, rm = divmod(
                    rm,
                    60
                )

                bar_len = 20

                filled = int(
                    bar_len
                    * step
                    / max_steps
                )

                bar = (
                    "█" * filled
                    + "░" * (
                        bar_len - filled
                    )
                )

                print(
                    f"\r  {bar} "
                    f"{pct:5.1f}% | "
                    f"Step {step}/{max_steps} | "
                    f"Loss {avg_loss:.4f} | "
                    f"{it_per_sec:.1f}it/s | "
                    f"{h}:{m:02d}:{s:02d} elapsed | "
                    f"ETA {rh}:{rm:02d}:{rs:02d}  ",
                    end="",
                    flush=True
                )

                step_start = time.time()
                loss_sum = 0.0
                loss_count = 0

            if (
                is_main
                and step % train_config.save_interval == 0
            ):
                state_dict = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )

                torch.save(
                    {
                        "model": state_dict,
                        "config": model_config,
                        "tokenizer": tokenizer_name,
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    os.path.join(
                        CKPT_DIR,
                        "latest.pt"
                    ),
                )

            if step >= max_steps:
                break

    if is_main:
        save_path = os.path.join(
            CKPT_DIR,
            "best.pt"
        )

        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        torch.save(
            {
                "model": state_dict,
                "config": model_config,
                "tokenizer": tokenizer_name,
            },
            save_path,
        )

        save_tokenizer(
            tokenizer,
            os.path.join(
                CKPT_DIR,
                "tokenizer.json"
            ),
        )

        elapsed = time.time() - start

        m, s = divmod(
            int(elapsed),
            60
        )

        h, m = divmod(
            m,
            60
        )

        final_loss = (
            loss.item()
            if "loss" in locals()
            else 0.0
        )

        print(
            f"\n\nDone! "
            f"{h}:{m:02d}:{s:02d} | "
            f"Final loss: {final_loss:.4f}",
            flush=True
        )

        print(
            f"Saved to {save_path}",
            flush=True
        )

    if use_ddp:
        dist.destroy_process_group()


# ============================================================
# TPU / PyTorch XLA training
# ============================================================

def train_tpu(index):
    """
    Training worker for one TPU core.

    v5e-8 uses eight TPU devices/processes.
    Each process owns one XLA device and one model replica.
    """

    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    device = torch_xla.device()

    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    is_master = xm.is_master_ordinal()

    # --------------------------------------------------------
    # TPU information
    # --------------------------------------------------------

    if is_master:
        print(
            "\n========================================",
            flush=True
        )

        print(
            "TPU TRAINING",
            flush=True
        )

        print(
            f"PyTorch: {torch.__version__}",
            flush=True
        )

        print(
            f"PyTorch/XLA device: {device}",
            flush=True
        )

        print(
            f"TPU processes/devices: {world_size}",
            flush=True
        )

        print(
            "Expected for v5e-8: 8",
            flush=True
        )

        print(
            "Precision: BF16 autocast",
            flush=True
        )

        print(
            "========================================\n",
            flush=True
        )

    # --------------------------------------------------------
    # Build tokenizer/model
    # --------------------------------------------------------

    model_config = auto_config_from_data(
        DATA_FILE
    )

    train_config = auto_train_config(
        DATA_FILE,
        CKPT_DIR
    )

    tokenizer_name = "word"

    tokenizer = get_tokenizer(
        tokenizer_name,
        data_file=DATA_FILE
    )

    if is_master:
        print(
            f"Tokenizer: {tokenizer_name}",
            flush=True
        )

        print(
            f"Vocab size: {tokenizer.vocab_size}",
            flush=True
        )

        print(
            tokenizer.vocab_info(),
            flush=True
        )

    dataset = TextDataset(
        DATA_FILE,
        tokenizer,
        model_config.max_seq_len
    )

    if len(dataset) == 0:
        if is_master:
            print(
                "ERROR: No valid <q>/<a> pairs found!",
                flush=True
            )

        return

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_config
    ).to(device)

    model.train()

    print_model_information(
        model,
        model_config,
        tokenizer,
        device,
        rank,
        world_size,
        "TPU v5e",
    )

    # --------------------------------------------------------
    # DataLoader
    #
    # The regular DataLoader stays on the CPU.
    # MpDeviceLoader asynchronously moves batches to XLA.
    # --------------------------------------------------------

    batch_size = min(
        train_config.batch_size,
        len(dataset)
    )

    # Every TPU process receives its own portion of the
    # effective global batch through the XLA input pipeline.
    #
    # Keep the same configured batch size as the original
    # training code. The XLA process topology handles the
    # replicated execution.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(
            4,
            os.cpu_count() or 1
        ),
        pin_memory=False,
        persistent_workers=(
            (os.cpu_count() or 1) > 1
        ),
        prefetch_factor=2,
        drop_last=True,
    )

    train_loader = pl.MpDeviceLoader(
        loader,
        device
    )

    if is_master:
        print(
            f"Dataset: {len(dataset)} samples",
            flush=True
        )

        print(
            f"Host batches: {len(loader)}",
            flush=True
        )

        print(
            "Using PyTorch/XLA MpDeviceLoader",
            flush=True
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        train_config.max_steps
    )

    os.makedirs(
        CKPT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Checkpoint resume
    # --------------------------------------------------------

    max_steps = train_config.max_steps
    start_step = 0

    resume_path = os.path.join(
        CKPT_DIR,
        "latest.pt"
    )

    if (
        "--resume" in sys.argv
        and os.path.exists(resume_path)
    ):
        if is_master:
            print(
                f"Loading checkpoint: {resume_path}",
                flush=True
            )

        # Load checkpoint on CPU first.
        # This avoids requiring the checkpoint itself to
        # already contain XLA tensors.
        ckpt = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler"]
        )

        start_step = int(
            ckpt["step"]
        )

        if is_master:
            print(
                f"Resumed from step {start_step}",
                flush=True
            )

    # Make sure every TPU process starts from the same
    # checkpoint state before training continues.
    xm.rendezvous(
        "checkpoint_loaded"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    step = start_step

    start = time.time()

    loss_sum = 0.0
    loss_count = 0

    step_start = time.time()

    # XLA BF16 autocast.
    #
    # Parameters remain FP32 while supported operations are
    # executed in BF16 where appropriate.
    use_bfloat16 = True

    while step < max_steps:

        for x, y, mask in train_loader:

            # ------------------------------------------------
            # XLA tensors already live on the TPU because
            # MpDeviceLoader handles the transfer.
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            if use_bfloat16:
                autocast_context = torch.autocast(
                    device_type="xla",
                    dtype=torch.bfloat16,
                )
            else:
                autocast_context = torch.autocast(
                    device_type="xla",
                    enabled=False,
                )

            with autocast_context:

                logits, loss = model(
                    x,
                    y,
                    loss_mask=mask
                )

                loss = loss.mean()

            loss.backward()

            # XLA optimizer step performs the required
            # XLA execution/synchronization for the optimizer.
            xm.optimizer_step(
                optimizer,
                barrier=True
            )

            scheduler.step()

            step += 1

            # ------------------------------------------------
            # Getting .item() forces an XLA synchronization.
            # Only do it at logging intervals to avoid adding
            # a synchronization to every single step.
            # ------------------------------------------------

            if (
                step % train_config.log_interval == 0
            ):
                loss_value = (
                    loss.detach()
                    .float()
                    .item()
                )

                loss_sum += loss_value
                loss_count += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                is_master
                and step % train_config.log_interval == 0
            ):
                elapsed = time.time() - start

                it_per_sec = (
                    loss_count
                    / max(
                        0.001,
                        time.time() - step_start
                    )
                )

                avg_loss = (
                    loss_sum
                    / max(1, loss_count)
                )

                pct = (
                    step
                    / max_steps
                    * 100
                )

                remaining = (
                    max_steps - step
                ) / max(
                    0.001,
                    it_per_sec
                )

                m, s = divmod(
                    int(elapsed),
                    60
                )

                h, m = divmod(
                    m,
                    60
                )

                rm, rs = divmod(
                    int(remaining),
                    60
                )

                rh, rm = divmod(
                    rm,
                    60
                )

                bar_len = 20

                filled = int(
                    bar_len
                    * step
                    / max_steps
                )

                bar = (
                    "█" * filled
                    + "░" * (
                        bar_len - filled
                    )
                )

                print(
                    f"\r  {bar} "
                    f"{pct:5.1f}% | "
                    f"Step {step}/{max_steps} | "
                    f"Loss {avg_loss:.4f} | "
                    f"{it_per_sec:.2f}it/s | "
                    f"{h}:{m:02d}:{s:02d} elapsed | "
                    f"ETA {rh}:{rm:02d}:{rs:02d}  ",
                    end="",
                    flush=True
                )

                step_start = time.time()

                loss_sum = 0.0
                loss_count = 0

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                step % train_config.save_interval == 0
            ):
                # Synchronize all TPU workers before saving.
                xm.rendezvous(
                    f"checkpoint_{step}"
                )

                if is_master:

                    # Move model weights to CPU for a portable
                    # checkpoint that can later be loaded by
                    # normal PyTorch/GPU inference.
                    state_dict = {
                        key: value.detach().cpu()
                        for key, value
                        in model.state_dict().items()
                    }

                    checkpoint = {
                        "model": state_dict,
                        "config": model_config,
                        "tokenizer": tokenizer_name,
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "backend": "xla",
                        "precision": "bf16",
                    }

                    checkpoint_path = os.path.join(
                        CKPT_DIR,
                        "latest.pt"
                    )

                    # xm.save is designed for XLA-aware
                    # distributed checkpoint saving.
                    xm.save(
                        checkpoint,
                        checkpoint_path
                    )

                    print(
                        f"\nCheckpoint saved at step {step}",
                        flush=True
                    )

                xm.rendezvous(
                    f"checkpoint_saved_{step}"
                )

            if step >= max_steps:
                break

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    xm.rendezvous(
        "final_checkpoint"
    )

    if is_master:

        save_path = os.path.join(
            CKPT_DIR,
            "best.pt"
        )

        state_dict = {
            key: value.detach().cpu()
            for key, value
            in model.state_dict().items()
        }

        final_checkpoint = {
            "model": state_dict,
            "config": model_config,
            "tokenizer": tokenizer_name,
            "step": step,
            "backend": "xla",
            "precision": "bf16",
        }

        xm.save(
            final_checkpoint,
            save_path
        )

        save_tokenizer(
            tokenizer,
            os.path.join(
                CKPT_DIR,
                "tokenizer.json"
            ),
        )

        elapsed = time.time() - start

        m, s = divmod(
            int(elapsed),
            60
        )

        h, m = divmod(
            m,
            60
        )

        final_loss = (
            loss.detach()
            .float()
            .item()
            if "loss" in locals()
            else 0.0
        )

        print(
            f"\n\nDone! "
            f"{h}:{m:02d}:{s:02d} | "
            f"Final loss: {final_loss:.4f}",
            flush=True
        )

        print(
            f"Saved to {save_path}",
            flush=True
        )

    xm.rendezvous(
        "training_finished"
    )


# ============================================================
# TPU launcher
# ============================================================

def launch_tpu():
    """
    Launch eight TPU workers for a v5e-8.

    PyTorch/XLA's multiprocessing launcher creates one
    training process per TPU device.
    """

    import torch_xla.distributed.xla_multiprocessing as xmp

    # v5e-8 = 8 TPU devices.
    num_tpu_cores = 8

    print(
        "Launching PyTorch/XLA across "
        f"{num_tpu_cores} TPU cores...",
        flush=True
    )

    xmp.spawn(
        train_tpu,
        args=(),
        nprocs=num_tpu_cores,
        start_method="fork",
    )


# ============================================================
# Entry point
# ============================================================

def main():

    use_tpu = "--tpu" in sys.argv

    if use_tpu:
        # Remove the launcher flag before passing sys.argv
        # into the worker code.
        sys.argv = [
            arg
            for arg in sys.argv
            if arg != "--tpu"
        ]

        launch_tpu()
        return

    # Otherwise use the original GPU/CPU path.
    train_gpu()


if __name__ == "__main__":

    # TPU path must not go through the CUDA auto-launcher.
    if "--tpu" not in sys.argv:
        auto_launch_gpu()

    main()
