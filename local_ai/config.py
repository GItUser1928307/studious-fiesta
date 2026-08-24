import json
import os
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    intermediate_size: int = 1536
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def total_params(self):
        embed = self.vocab_size * self.hidden_size
        per_block_qkv = self.hidden_size * (3 * self.hidden_size)
        per_block_attn_out = self.hidden_size * self.hidden_size
        per_block_gate = self.hidden_size * self.intermediate_size
        per_block_up = self.hidden_size * self.intermediate_size
        per_block_down = self.intermediate_size * self.hidden_size
        per_block_norm = self.hidden_size * 2

        per_block = (
            per_block_qkv
            + per_block_attn_out
            + per_block_gate
            + per_block_up
            + per_block_down
            + per_block_norm
        )

        total = embed + self.num_layers * per_block + self.hidden_size
        return total

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class TrainConfig:
    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000

    # ---------------------------------------------------------
    # Logging / evaluation / checkpoints
    # ---------------------------------------------------------
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    gradient_clip: float = 1.0

    save_dir: str = "checkpoints"
    data_file: str = "data.txt"
    tokenizer_name: str = "word"

    # ---------------------------------------------------------
    # Hardware / precision
    # ---------------------------------------------------------
    # "auto" lets the training script choose the available device.
    # Supported values should be handled by the training script:
    #   auto / cpu / cuda / xla
    device: str = "auto"

    # TPU/XLA training uses BF16 when enabled.
    # Keep this false by default so the same config remains usable
    # with CPU/GPU training.
    tpu_bfloat16: bool = True

    # Enable XLA compilation when running on TPU.
    xla_compile: bool = True

    # Number of TPU processes/cores.
    # v5e-8 = 8 TPU cores.
    tpu_cores: int = 8

    # ---------------------------------------------------------
    # Distributed training
    # ---------------------------------------------------------
    distributed: bool = False

    # Automatically use DistributedDataParallel when supported.
    use_ddp: bool = True

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------
    # Gradient accumulation lets you increase the effective batch
    # size without requiring the entire batch to fit in memory.
    gradient_accumulation_steps: int = 1

    # XLA can benefit from keeping tensor shapes consistent.
    # The training/data pipeline should avoid dynamically changing
    # sequence lengths when this is enabled.
    static_shapes: bool = True


SMALL_CONFIG = ModelConfig(
    vocab_size=98,
    hidden_size=128,
    num_layers=6,
    num_heads=2,
    intermediate_size=512,
    max_seq_len=128,
)


FULL_CONFIG = ModelConfig()


def auto_config():
    return ModelConfig(
        vocab_size=99,
        hidden_size=48,
        num_layers=6,
        num_heads=2,
        intermediate_size=96,
        max_seq_len=32,
    )


def auto_config_from_data(data_file: str, max_seq_len: int = 96):
    from tokenizer import WordTokenizer

    tok = WordTokenizer.build(data_file)
    vocab_size = tok.vocab_size

    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=22,
        num_heads=8,
        intermediate_size=1408,
        max_seq_len=max_seq_len,
    )


def auto_train_config(
    data_file="quick_train_data.txt",
    save_dir="quick_ckpt",
):
    return TrainConfig(
        batch_size=32,
        max_steps=300000,
        learning_rate=3e-4,
        log_interval=1,
        save_interval=200,
        data_file=data_file,
        save_dir=save_dir,

        # TPU-friendly defaults
        device="auto",
        tpu_bfloat16=True,
        xla_compile=True,
        tpu_cores=8,
        distributed=True,
        use_ddp=True,
        gradient_accumulation_steps=1,
        static_shapes=True,
    )
