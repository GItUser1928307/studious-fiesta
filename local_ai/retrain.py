#!/usr/bin/env python3
"""
Train the model on CPU, NVIDIA CUDA GPUs, or TPU v5e-8 via PyTorch/XLA.

GPU / CPU:
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

DATA_FILE = os.path.join(
    PARENT_DIR,
    "quick_train_data.txt",
)

CKPT_DIR = os.path.join(
    PARENT_DIR,
    "quick_ckpt",
)


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):

    def __init__(
        self,
        file_path,
        tokenizer,
        seq_len,
    ):
        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            lines = f.readlines()

        self.samples = []
        self.seq_len = seq_len

        i = 0

        while i < len(lines) - 1:

            q_line = lines[i].strip()
            a_line = lines[i + 1].strip()

            if (
                q_line.startswith("<q>")
                and a_line.startswith("<a>")
            ):

                q_tokens = [
                    token
                    for token in tokenizer.tokenize(q_line)
                    if token not in ("<q>", "<a>")
                ]

                a_tokens = [
                    token
                    for token in tokenizer.tokenize(a_line)
                    if token not in ("<q>", "<a>")
                ]

                input_ids = [
                    tokenizer.bos_token_id,
                    tokenizer.q_token_id,
                ]

                input_ids += [
                    tokenizer.word_to_id.get(
                        token,
                        tokenizer.unk_token_id,
                    )
                    for token in q_tokens
                ]

                input_ids.append(
                    tokenizer.a_token_id
                )

                target_ids = [
                    tokenizer.word_to_id.get(
                        token,
                        tokenizer.unk_token_id,
                    )
                    for token in a_tokens
                ]

                target_ids.append(
                    tokenizer.eos_token_id
                )

                full = input_ids + target_ids

                # We need one extra token because training uses
                # full[:-1] -> full[1:].
                if len(full) <= seq_len + 1:

                    x = torch.zeros(
                        seq_len,
                        dtype=torch.long,
                    )

                    y = torch.zeros(
                        seq_len,
                        dtype=torch.long,
                    )

                    mask = torch.zeros(
                        seq_len,
                        dtype=torch.bool,
                    )

                    n = len(full) - 1

                    x[:n] = torch.tensor(
                        full[:-1],
                        dtype=torch.long,
                    )

                    y[:n] = torch.tensor(
                        full[1:],
                        dtype=torch.long,
                    )

                    # Start calculating loss when the first
                    # answer token is predicted.
                    #
                    # The A-token itself is the input whose
                    # following target is the first answer token.
                    answer_start = len(input_ids) - 1

                    mask[
                        answer_start:n
                    ] = True

                    self.samples.append(
                        (
                            x,
                            y,
                            mask,
                        )
                    )

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

def build_scheduler(
    optimizer,
    max_steps,
):
    warmup_steps = min(
        100,
        max_steps // 10,
    )

    def lr_lambda(step):

        if step < warmup_steps:
            return step / max(
                1,
                warmup_steps,
            )

        progress = (
            step - warmup_steps
        ) / max(
            1,
            max_steps - warmup_steps,
        )

        progress = min(
            1.0,
            max(
                0.0,
                progress,
            ),
        )

        return (
            0.1
            + 0.9
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


def create_dataset_and_tokenizer(
    model_config,
):
    tokenizer_name = "word"

    tokenizer = get_tokenizer(
        tokenizer_name,
        data_file=DATA_FILE,
    )

    dataset = TextDataset(
        DATA_FILE,
        tokenizer,
        model_config.max_seq_len,
    )

    return (
        tokenizer_name,
        tokenizer,
        dataset,
    )


def _unwrap_for_checkpoint(model):
    """
    Unwrap DDP and torch.compile wrappers to get the raw
    eager model for checkpoint compatibility.

    benchmark.py uses the same unwrapping so resume works
    identically with and without --compile.
    """
    m = model
    # Unwrap DDP
    if hasattr(m, "module"):
        m = m.module
    # Unwrap torch.compile (OptimizedModule -> _orig_mod)
    # Handle nested wrappers
    for _ in range(3):
        if hasattr(m, "_orig_mod"):
            try:
                m = m._orig_mod
            except Exception:
                break
        else:
            break
    return m


def get_model_state_dict(model):
    """
    Return a normal CPU-compatible state dict.

    Works with regular, DDP, and compiled models.
    """
    inner = _unwrap_for_checkpoint(model)
    return inner.state_dict()


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
        flush=True,
    )

    print(
        f"Config: "
        f"hidden={model_config.hidden_size}, "
        f"layers={model_config.num_layers}, "
        f"heads={model_config.num_heads}, "
        f"head_dim={model_config.head_dim}, "
        f"intermediate={model_config.intermediate_size}, "
        f"seq={model_config.max_seq_len}, "
        f"vocab={model_config.vocab_size}, "
        f"rope_theta={model_config.rope_theta}, "
        f"tie_embeddings={model_config.tie_embeddings}",
        flush=True,
    )

    print(
        f"Device: {device}",
        flush=True,
    )

    if accelerator_name:
        print(
            f"Accelerator: {accelerator_name}",
            flush=True,
        )

    print(
        f"World size: {world_size}",
        flush=True,
    )

    print(
        "Architecture: "
        "RMSNorm + RoPE + SwiGLU + causal attention",
        flush=True,
    )

    print(
        "Tokenizer: word",
        flush=True,
    )

    print(
        f"Vocab size: {tokenizer.vocab_size}",
        flush=True,
    )

    print(
        tokenizer.vocab_info(),
        flush=True,
    )


def save_checkpoint(
    path,
    model,
    model_config,
    tokenizer_name,
    step,
    optimizer=None,
    scheduler=None,
    scaler=None,
    backend=None,
    precision=None,
):
    checkpoint = {
        "model": get_model_state_dict(model),
        "config": model_config,
        "tokenizer": tokenizer_name,
        "step": step,
    }

    if optimizer is not None:
        checkpoint["optimizer"] = (
            optimizer.state_dict()
        )

    if scheduler is not None:
        checkpoint["scheduler"] = (
            scheduler.state_dict()
        )

    if scaler is not None:
        checkpoint["scaler"] = (
            scaler.state_dict()
        )

    if backend is not None:
        checkpoint["backend"] = backend

    if precision is not None:
        checkpoint["precision"] = precision

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# CUDA / GPU training
# ============================================================

def auto_launch_gpu():
    """
    Preserve multi-GPU behavior.

    Running:

        python retrain.py

    automatically launches torchrun when multiple
    CUDA GPUs are detected.
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
        flush=True,
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
                str(i)
                for i in range(num_gpus)
            ),
        },
    )

    sys.exit(
        proc.returncode
    )


def train_gpu():

    import torch.distributed as dist
    from torch.nn.parallel import (
        DistributedDataParallel as DDP,
    )
    from torch.utils.data.distributed import (
        DistributedSampler,
    )

    os.environ.setdefault(
        "NCCL_IB_DISABLE",
        "1",
    )

    os.environ.setdefault(
        "NCCL_NET_GDR_LEVEL",
        "0",
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # --------------------------------------------------------
    # CLI overrides for Kaggle benchmarking
    # --------------------------------------------------------
    def _get_cli_arg(name, typ=int, default=None):
        if name in sys.argv:
            try:
                idx = sys.argv.index(name)
                return typ(sys.argv[idx + 1])
            except Exception:
                return default
        return default

    cli_batch_size = _get_cli_arg("--batch-size", int, None)
    cli_workers = _get_cli_arg("--workers", int, None)
    cli_prefetch = _get_cli_arg("--prefetch", int, None)
    cli_compile = "--compile" in sys.argv
    cli_no_compile = "--no-compile" in sys.argv
    cli_no_amp = "--no-amp" in sys.argv
    # --benchmark is handled in the training loop for extra timing

    use_ddp = "RANK" in os.environ

    if use_ddp:

        backend = (
            "nccl"
            if torch.cuda.is_available()
            else "gloo"
        )

        dist.init_process_group(
            backend
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                "0",
            )
        )

        if torch.cuda.is_available():

            device = torch.device(
                f"cuda:{local_rank}"
            )

            torch.cuda.set_device(
                device
            )

        else:
            device = torch.device(
                "cpu"
            )

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
            flush=True,
        )

        for i in range(num_gpus):

            print(
                f"  GPU {i}: "
                f"{torch.cuda.get_device_name(i)}",
                flush=True,
            )

        if use_ddp:

            print(
                f"Using DDP across "
                f"{world_size} processes",
                flush=True,
            )

        elif num_gpus == 1:

            print(
                f"Using single GPU: "
                f"{torch.cuda.get_device_name(0)}",
                flush=True,
            )

        else:

            print(
                "No GPU detected, using CPU",
                flush=True,
            )

    # --------------------------------------------------------
    # Configuration / tokenizer / dataset
    # --------------------------------------------------------

    model_config = auto_config_from_data(
        DATA_FILE
    )

    train_config = auto_train_config(
        DATA_FILE,
        CKPT_DIR,
    )

    # Apply CLI overrides (preserve checkpoint compatibility)
    if cli_batch_size is not None:
        train_config.batch_size = cli_batch_size
    if cli_workers is not None:
        train_config.num_workers = cli_workers
    if cli_prefetch is not None:
        train_config.prefetch_factor = cli_prefetch
    if cli_compile:
        train_config.compile_enabled = True
    if cli_no_compile:
        train_config.compile_enabled = False
    if cli_no_amp:
        train_config.amp_enabled = False

    (
        tokenizer_name,
        tokenizer,
        dataset,
    ) = create_dataset_and_tokenizer(
        model_config
    )

    if is_main:

        print(
            f"Tokenizer: {tokenizer_name}",
            flush=True,
        )

        print(
            f"Vocab size: "
            f"{tokenizer.vocab_size}",
            flush=True,
        )

        print(
            tokenizer.vocab_info(),
            flush=True,
        )

    if len(dataset) == 0:

        if is_main:

            print(
                "ERROR: No valid "
                "<q>/<a> pairs found "
                "in dataset!",
                flush=True,
            )

        if use_ddp:
            dist.destroy_process_group()

        return

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_config
    ).to(device)

    # torch.compile for 2×T4 — test on Kaggle, keep only if faster.
    # Compiled before DDP gives best compatibility with PyTorch 2.10.
    if train_config.compile_enabled and device.type == "cuda":
        try:
            model = torch.compile(
                model,
                mode=train_config.compile_mode,
            )
            if is_main:
                print(
                    f"torch.compile enabled "
                    f"(mode={train_config.compile_mode})",
                    flush=True,
                )
        except Exception as e:
            if is_main:
                print(
                    f"torch.compile failed: {e} "
                    f"— continuing without compilation",
                    flush=True,
                )

    if use_ddp:

        model = DDP(
            model,
            device_ids=(
                [local_rank]
                if device.type == "cuda"
                else None
            ),
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    model_for_count = _unwrap_for_checkpoint(model)

    print_model_information(
        model_for_count,
        model_config,
        tokenizer,
        device,
        rank,
        world_size,
        "NVIDIA CUDA"
        if device.type == "cuda"
        else "CPU",
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    batch_size = min(
        train_config.batch_size,
        len(dataset),
    )

    sampler = (
        DistributedSampler(
            dataset,
            shuffle=True,
        )
        if use_ddp
        else None
    )

    # Tuned for Kaggle 2×T4: 4 vCPUs total, 2 processes.
    # 2 workers per process avoids oversubscription.
    requested_workers = getattr(
        train_config, "num_workers", 2
    )
    num_workers = min(
        requested_workers,
        os.cpu_count() or 1,
    )
    # Cap at 2 per GPU when using DDP
    if use_ddp and world_size > 1:
        num_workers = min(num_workers, 2)
    if num_workers < 0:
        num_workers = 0

    prefetch = getattr(
        train_config, "prefetch_factor", 2
    )
    use_pin = getattr(
        train_config, "pin_memory", True
    ) and device.type == "cuda"

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": use_pin,
        "drop_last": True,
    }

    if num_workers > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = prefetch

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    if is_main:

        print(
            f"Dataset: {len(dataset)} samples, "
            f"{len(loader)} batches",
            flush=True,
        )
        print(
            f"DataLoader: workers={num_workers}, "
            f"prefetch={prefetch}, "
            f"pin_memory={use_pin}, "
            f"batch={batch_size} "
            f"(effective {batch_size * world_size}), "
            f"drop_last=True",
            flush=True,
        )

    # --------------------------------------------------------
    # Mixed precision — FP16 for T4 Tensor Cores
    # --------------------------------------------------------

    use_amp = (
        getattr(train_config, "amp_enabled", True)
        and device.type == "cuda"
    )

    amp_dtype = torch.float16
    # Allow override via config but force fp16 for T4
    cfg_dtype = getattr(
        train_config, "amp_dtype", "fp16"
    )
    if cfg_dtype == "bf16":
        # T4 does not have BF16 Tensor Cores; keep FP16
        if is_main:
            print(
                "Warning: BF16 requested but T4 uses FP16 — "
                "forcing fp16",
                flush=True,
            )
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    if is_main:
        print(
            f"AMP: {'fp16' if use_amp else 'fp32'} "
            f"(dtype={amp_dtype})",
            flush=True,
        )

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        train_config.max_steps,
    )

    os.makedirs(
        CKPT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    max_steps = train_config.max_steps
    start_step = 0

    resume_path = os.path.join(
        CKPT_DIR,
        "latest.pt",
    )

    if (
        "--resume" in sys.argv
        and os.path.exists(resume_path)
    ):

        if is_main:

            print(
                f"Loading checkpoint: "
                f"{resume_path}",
                flush=True,
            )

        ckpt = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        # Use unwrapped model so compiled and eager checkpoints are compatible.
        # This mirrors benchmark.py's unwrapping and guarantees
        # existing checkpoints load without restarting.
        model_to_load = _unwrap_for_checkpoint(model)

        try:
            model_to_load.load_state_dict(
                ckpt["model"],
                strict=True,
            )
        except Exception as e:
            # Compiled wrapper can expose _orig_mod state dict mismatch.
            # Fall back to eager: disable compile for this run and retry
            # with the unwrapped eager model so training continues.
            if getattr(train_config, "compile_enabled", False):
                if is_main:
                    print(
                        f"Compiled resume failed ({e}) — "
                        f"falling back to eager model",
                        flush=True,
                    )
                train_config.compile_enabled = False
                # The unwrapped eager model is already model_to_load;
                # retry loading (should succeed for any valid checkpoint)
                model_to_load.load_state_dict(
                    ckpt["model"],
                    strict=True,
                )
                if is_main:
                    print(
                        "Fallback to eager succeeded — "
                        "continuing without compilation",
                        flush=True,
                    )
            else:
                raise

        if "optimizer" in ckpt:
            optimizer.load_state_dict(
                ckpt["optimizer"]
            )

        if "scheduler" in ckpt:
            scheduler.load_state_dict(
                ckpt["scheduler"]
            )

        if (
            "scaler" in ckpt
            and use_amp
        ):
            scaler.load_state_dict(
                ckpt["scaler"]
            )

        start_step = int(
            ckpt.get(
                "step",
                0,
            )
        )

        if is_main:

            print(
                f"Resumed from step "
                f"{start_step}",
                flush=True,
            )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    step = start_step

    start = time.time()

    # Use tensor accumulation on GPU to avoid per-step sync
    interval_losses = []
    loss_count = 0

    step_start = time.time()
    epoch = start_step // max(1, len(loader))

    while step < max_steps:

        if sampler is not None:
            sampler.set_epoch(epoch)

        for x, y, mask in loader:

            x = x.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            y = y.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            mask = mask.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):

                logits, loss = model(
                    x,
                    y,
                    loss_mask=mask,
                )

                loss = loss.mean()

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.gradient_clip,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            step += 1

            # Accumulate without sync — only .item() at log interval
            if is_main:
                interval_losses.append(loss.detach())
            loss_count += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                is_main
                and step
                % train_config.log_interval
                == 0
            ):

                elapsed = (
                    time.time()
                    - start
                )

                interval_time = (
                    time.time()
                    - step_start
                )

                it_per_sec = (
                    loss_count
                    / max(
                        0.001,
                        interval_time,
                    )
                )

                tokens_per_sec = (
                    it_per_sec
                    * batch_size
                    * model_config.max_seq_len
                    * world_size
                )
                samples_per_sec = (
                    it_per_sec * batch_size * world_size
                )

                if interval_losses:
                    try:
                        avg_loss = (
                            torch.stack(interval_losses)
                            .float()
                            .mean()
                            .item()
                        )
                    except Exception:
                        avg_loss = 0.0
                else:
                    avg_loss = 0.0

                pct = (
                    step
                    / max_steps
                    * 100
                )

                remaining = (
                    max_steps - step
                ) / max(
                    0.001,
                    it_per_sec,
                )

                m, s = divmod(
                    int(elapsed),
                    60,
                )

                h, m = divmod(
                    m,
                    60,
                )

                rm, rs = divmod(
                    int(remaining),
                    60,
                )

                rh, rm = divmod(
                    rm,
                    60,
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
                        bar_len
                        - filled
                    )
                )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                print(
                    f"\r  {bar} "
                    f"{pct:5.1f}% | "
                    f"Step {step}/{max_steps} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {current_lr:.2e} | "
                    f"{it_per_sec:.1f}it/s "
                    f"{tokens_per_sec:.0f}tok/s | "
                    f"{h}:{m:02d}:{s:02d} elapsed | "
                    f"ETA {rh}:{rm:02d}:{rs:02d}  ",
                    end="",
                    flush=True,
                )

                step_start = time.time()

                interval_losses = []
                loss_count = 0

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                is_main
                and step
                % train_config.save_interval
                == 0
            ):

                save_checkpoint(
                    os.path.join(
                        CKPT_DIR,
                        "latest.pt",
                    ),
                    model,
                    model_config,
                    tokenizer_name,
                    step,
                    optimizer,
                    scheduler,
                    scaler,
                    backend=(
                        "cuda"
                        if device.type == "cuda"
                        else "cpu"
                    ),
                    precision=(
                        "fp16"
                        if use_amp
                        else "fp32"
                    ),
                )

                print(
                    f"\nCheckpoint saved "
                    f"at step {step}",
                    flush=True,
                )

            if step >= max_steps:
                break

        epoch += 1

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    if is_main:

        save_path = os.path.join(
            CKPT_DIR,
            "best.pt",
        )

        save_checkpoint(
            save_path,
            model,
            model_config,
            tokenizer_name,
            step,
            backend=(
                "cuda"
                if device.type == "cuda"
                else "cpu"
            ),
            precision=(
                "fp16"
                if use_amp
                else "fp32"
            ),
        )

        save_tokenizer(
            tokenizer,
            os.path.join(
                CKPT_DIR,
                "tokenizer.json",
            ),
        )

        elapsed = time.time() - start

        m, s = divmod(
            int(elapsed),
            60,
        )

        h, m = divmod(
            m,
            60,
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
            flush=True,
        )

        print(
            f"Saved to {save_path}",
            flush=True,
        )

    if use_ddp:
        dist.destroy_process_group()


# ============================================================
# TPU / PyTorch XLA training
# ============================================================

def train_tpu(index):

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
            flush=True,
        )

        print(
            "TPU TRAINING",
            flush=True,
        )

        print(
            f"PyTorch: {torch.__version__}",
            flush=True,
        )

        print(
            f"PyTorch/XLA device: {device}",
            flush=True,
        )

        print(
            f"TPU processes/devices: "
            f"{world_size}",
            flush=True,
        )

        print(
            "Expected for v5e-8: 8",
            flush=True,
        )

        print(
            "Precision: BF16 autocast",
            flush=True,
        )

        print(
            "Architecture: "
            "RMSNorm + RoPE + SwiGLU",
            flush=True,
        )

        print(
            "========================================\n",
            flush=True,
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    model_config = auto_config_from_data(
        DATA_FILE
    )

    train_config = auto_train_config(
        DATA_FILE,
        CKPT_DIR,
    )

    tokenizer_name = "word"

    tokenizer = get_tokenizer(
        tokenizer_name,
        data_file=DATA_FILE,
    )

    if is_master:

        print(
            f"Tokenizer: {tokenizer_name}",
            flush=True,
        )

        print(
            f"Vocab size: "
            f"{tokenizer.vocab_size}",
            flush=True,
        )

        print(
            tokenizer.vocab_info(),
            flush=True,
        )

    dataset = TextDataset(
        DATA_FILE,
        tokenizer,
        model_config.max_seq_len,
    )

    if len(dataset) == 0:

        if is_master:

            print(
                "ERROR: No valid "
                "<q>/<a> pairs found!",
                flush=True,
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
    # --------------------------------------------------------

    batch_size = min(
        train_config.batch_size,
        len(dataset),
    )

    cpu_workers = min(
        4,
        os.cpu_count() or 1,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": cpu_workers,
        "pin_memory": False,
        "drop_last": True,
    }

    if cpu_workers > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    train_loader = pl.MpDeviceLoader(
        loader,
        device,
    )

    if is_master:

        print(
            f"Dataset: {len(dataset)} samples",
            flush=True,
        )

        print(
            f"Host batches: {len(loader)}",
            flush=True,
        )

        print(
            "Using PyTorch/XLA "
            "MpDeviceLoader",
            flush=True,
        )

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        train_config.max_steps,
    )

    os.makedirs(
        CKPT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    max_steps = train_config.max_steps
    start_step = 0

    resume_path = os.path.join(
        CKPT_DIR,
        "latest.pt",
    )

    if (
        "--resume" in sys.argv
        and os.path.exists(resume_path)
    ):

        if is_master:

            print(
                f"Loading checkpoint: "
                f"{resume_path}",
                flush=True,
            )

        ckpt = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model"],
            strict=True,
        )

        if "optimizer" in ckpt:

            optimizer.load_state_dict(
                ckpt["optimizer"]
            )

        if "scheduler" in ckpt:

            scheduler.load_state_dict(
                ckpt["scheduler"]
            )

        start_step = int(
            ckpt.get(
                "step",
                0,
            )
        )

        if is_master:

            print(
                f"Resumed from step "
                f"{start_step}",
                flush=True,
            )

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

    while step < max_steps:

        for x, y, mask in train_loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="xla",
                dtype=torch.bfloat16,
            ):

                logits, loss = model(
                    x,
                    y,
                    loss_mask=mask,
                )

                loss = loss.mean()

            loss.backward()

            # Gradient clipping is useful for the new
            # attention/SwiGLU architecture as well.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.gradient_clip,
            )

            xm.optimizer_step(
                optimizer,
                barrier=True,
            )

            scheduler.step()

            step += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                step
                % train_config.log_interval
                == 0
            ):

                loss_value = (
                    loss.detach()
                    .float()
                    .item()
                )

                loss_sum += loss_value
                loss_count += 1

            if (
                is_master
                and step
                % train_config.log_interval
                == 0
            ):

                elapsed = (
                    time.time()
                    - start
                )

                interval_time = (
                    time.time()
                    - step_start
                )

                it_per_sec = (
                    loss_count
                    / max(
                        0.001,
                        interval_time,
                    )
                )

                avg_loss = (
                    loss_sum
                    / max(
                        1,
                        loss_count,
                    )
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
                    it_per_sec,
                )

                m, s = divmod(
                    int(elapsed),
                    60,
                )

                h, m = divmod(
                    m,
                    60,
                )

                rm, rs = divmod(
                    int(remaining),
                    60,
                )

                rh, rm = divmod(
                    rm,
                    60,
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
                        bar_len
                        - filled
                    )
                )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                print(
                    f"\r  {bar} "
                    f"{pct:5.1f}% | "
                    f"Step {step}/{max_steps} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {current_lr:.2e} | "
                    f"{it_per_sec:.2f}it/s | "
                    f"{h}:{m:02d}:{s:02d} elapsed | "
                    f"ETA {rh}:{rm:02d}:{rs:02d}  ",
                    end="",
                    flush=True,
                )

                step_start = time.time()

                loss_sum = 0.0
                loss_count = 0

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                step
                % train_config.save_interval
                == 0
            ):

                xm.rendezvous(
                    f"checkpoint_{step}"
                )

                if is_master:

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
                        "optimizer": (
                            optimizer.state_dict()
                        ),
                        "scheduler": (
                            scheduler.state_dict()
                        ),
                        "backend": "xla",
                        "precision": "bf16",
                    }

                    checkpoint_path = os.path.join(
                        CKPT_DIR,
                        "latest.pt",
                    )

                    xm.save(
                        checkpoint,
                        checkpoint_path,
                    )

                    print(
                        f"\nCheckpoint saved "
                        f"at step {step}",
                        flush=True,
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
            "best.pt",
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
            save_path,
        )

        save_tokenizer(
            tokenizer,
            os.path.join(
                CKPT_DIR,
                "tokenizer.json",
            ),
        )

        elapsed = time.time() - start

        m, s = divmod(
            int(elapsed),
            60,
        )

        h, m = divmod(
            m,
            60,
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
            flush=True,
        )

        print(
            f"Saved to {save_path}",
            flush=True,
        )

    xm.rendezvous(
        "training_finished"
    )


# ============================================================
# TPU launcher
# ============================================================

def launch_tpu():

    import torch_xla.distributed.xla_multiprocessing as xmp

    num_tpu_cores = 8

    print(
        "Launching PyTorch/XLA across "
        f"{num_tpu_cores} TPU cores...",
        flush=True,
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

        sys.argv = [
            arg
            for arg in sys.argv
            if arg != "--tpu"
        ]

        launch_tpu()
        return

    train_gpu()


if __name__ == "__main__":

    if "--tpu" not in sys.argv:
        auto_launch_gpu()

    main()
