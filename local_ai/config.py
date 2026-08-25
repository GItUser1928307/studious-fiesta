import json
import os
from dataclasses import dataclass, asdict


# ============================================================
# Model configuration
# ============================================================

@dataclass
class ModelConfig:
    # IMPORTANT:
    # These architecture values are intentionally preserved.
    # Changing them makes existing model checkpoints incompatible.

    vocab_size: int = 50257

    hidden_size: int = 512
    num_layers: int = 22
    num_heads: int = 8
    intermediate_size: int = 1408

    max_seq_len: int = 96

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def total_params(self):
        """
        Calculate the expected number of model parameters.

        This calculation intentionally matches the architecture
        used by model.py.
        """

        # Token embedding.
        embed = self.vocab_size * self.hidden_size

        # Attention.
        per_block_qkv = (
            self.hidden_size
            * (3 * self.hidden_size)
        )

        per_block_attn_out = (
            self.hidden_size
            * self.hidden_size
        )

        # SwiGLU.
        per_block_gate = (
            self.hidden_size
            * self.intermediate_size
        )

        per_block_up = (
            self.hidden_size
            * self.intermediate_size
        )

        per_block_down = (
            self.intermediate_size
            * self.hidden_size
        )

        # Two RMSNorm layers.
        per_block_norm = (
            self.hidden_size * 2
        )

        per_block = (
            per_block_qkv
            + per_block_attn_out
            + per_block_gate
            + per_block_up
            + per_block_down
            + per_block_norm
        )

        # Final RMSNorm.
        final_norm = self.hidden_size

        total = (
            embed
            + self.num_layers * per_block
            + final_norm
        )

        # When embeddings are tied, lm_head shares the embedding
        # weight and therefore does NOT add another parameter matrix.
        #
        # If tie_embeddings were false, model.py would contain an
        # additional vocab_size * hidden_size parameter matrix.
        if not self.tie_embeddings:
            total += (
                self.vocab_size
                * self.hidden_size
            )

        return total

    def save(self, path):
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                asdict(self),
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with open(
            path,
            encoding="utf-8",
        ) as f:
            return cls(
                **json.load(f)
            )


# ============================================================
# Training configuration
# ============================================================

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

    # Supported values:
    #
    #   auto
    #   cpu
    #   cuda
    #   xla
    #
    # The training script decides which backend to actually use.
    device: str = "auto"

    # TPU/XLA BF16 support.
    tpu_bfloat16: bool = True

    # Enable XLA compilation when running on TPU.
    xla_compile: bool = True

    # TPU v5e-8 has 8 devices.
    tpu_cores: int = 8

    # ---------------------------------------------------------
    # Distributed training
    # ---------------------------------------------------------

    distributed: bool = False

    use_ddp: bool = True

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    gradient_accumulation_steps: int = 1

    static_shapes: bool = True

    # DataLoader tuning (Kaggle 2×T4: 4 vCPUs total).
    # num_workers is per-process; 2 is optimal to avoid
    # oversubscription with 2 DDP processes.
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Mixed precision — FP16 is optimal for T4 Tensor Cores.
    amp_enabled: bool = True
    amp_dtype: str = "fp16"

    # torch.compile — PyTorch 2.10, test on Kaggle before enabling.
    compile_enabled: bool = False
    compile_mode: str = "default"


# ============================================================
# Preset model configurations
# ============================================================

# Small configuration.
#
# Kept unchanged from the existing project so code that relies
# on SMALL_CONFIG continues to work.
SMALL_CONFIG = ModelConfig(
    vocab_size=98,
    hidden_size=128,
    num_layers=6,
    num_heads=2,
    intermediate_size=512,
    max_seq_len=128,
)


# Full configuration.
#
# IMPORTANT:
# This remains the 512 / 22 / 8 / 1408 architecture.
FULL_CONFIG = ModelConfig(
    vocab_size=50257,
    hidden_size=512,
    num_layers=22,
    num_heads=8,
    intermediate_size=1408,
    max_seq_len=96,
)


# ============================================================
# Automatic configuration
# ============================================================

def auto_config():
    """
    Return the project's fixed main architecture.

    IMPORTANT:
    Do not change the architecture values here.

    Existing long-running checkpoints depend on:
        hidden_size       = 512
        num_layers        = 22
        num_heads         = 8
        intermediate_size = 1408
        max_seq_len       = 96
    """

    return ModelConfig(
        vocab_size=99,
        hidden_size=512,
        num_layers=22,
        num_heads=8,
        intermediate_size=1408,
        max_seq_len=96,
    )


def auto_config_from_data(
    data_file: str,
    max_seq_len: int = 96,
):
    """
    Build the model configuration using the vocabulary from the
    training data while preserving the exact model architecture.

    The transformer dimensions are intentionally fixed.

    IMPORTANT:
    vocab_size is the only dimension that depends on the tokenizer.

    For RESUMING an existing checkpoint, the tokenizer vocabulary
    must remain compatible with the vocabulary used to create that
    checkpoint.
    """

    from tokenizer import WordTokenizer

    tok = WordTokenizer.build(
        data_file
    )

    vocab_size = tok.vocab_size

    return ModelConfig(
        vocab_size=vocab_size,

        # DO NOT CHANGE THESE.
        hidden_size=512,
        num_layers=22,
        num_heads=8,
        intermediate_size=1408,

        max_seq_len=max_seq_len,

        rope_theta=10000.0,
        rms_norm_eps=1e-6,

        tie_embeddings=True,
        dropout=0.0,
    )


# ============================================================
# Automatic training configuration
# ============================================================

def auto_train_config(
    data_file="quick_train_data.txt",
    save_dir="quick_ckpt",
):
    return TrainConfig(
        batch_size=32,

        max_steps=300000,

        learning_rate=3e-4,

        weight_decay=0.1,

        warmup_steps=100,

        log_interval=10,

        save_interval=200,

        eval_interval=100,

        gradient_clip=1.0,

        data_file=data_file,

        save_dir=save_dir,

        tokenizer_name="word",

        # -----------------------------------------------------
        # TPU-friendly defaults
        # -----------------------------------------------------

        device="auto",

        tpu_bfloat16=True,

        xla_compile=True,

        tpu_cores=8,

        distributed=True,

        use_ddp=True,

        gradient_accumulation_steps=1,

        static_shapes=True,

        # -----------------------------------------------------
        # GPU performance — tuned for 2× T4
        # -----------------------------------------------------
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        amp_enabled=True,
        amp_dtype="fp16",
        compile_enabled=False,
        compile_mode="default",
    )
