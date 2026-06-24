import json
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
        per_block = per_block_qkv + per_block_attn_out + per_block_gate + per_block_up + per_block_down + per_block_norm
        total = embed + self.num_layers * per_block + self.hidden_size
        return total

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    gradient_clip: float = 1.0
    save_dir: str = "checkpoints"
    data_file: str = "data.txt"


SMALL_CONFIG = ModelConfig(
    vocab_size=128,
    hidden_size=256,
    num_layers=6,
    num_heads=4,
    intermediate_size=1024,
    max_seq_len=256,
)

FULL_CONFIG = ModelConfig()


def auto_config():
    return ModelConfig(vocab_size=128, hidden_size=32, num_layers=3, num_heads=2, intermediate_size=80, max_seq_len=32)


def auto_train_config(data_file="quick_train_data.txt", save_dir="quick_ckpt"):
    from system import get_avail_ram_gb, get_cpu_threads
    avail = get_avail_ram_gb()
    threads = get_cpu_threads()
    if avail >= 3:
        return TrainConfig(batch_size=64, max_steps=500, learning_rate=5e-4, log_interval=25, save_interval=100, data_file=data_file, save_dir=save_dir)
    elif avail >= 1.5:
        return TrainConfig(batch_size=64, max_steps=300, learning_rate=5e-4, log_interval=25, save_interval=100, data_file=data_file, save_dir=save_dir)
    else:
        return TrainConfig(batch_size=32, max_steps=200, learning_rate=5e-4, log_interval=10, save_interval=50, data_file=data_file, save_dir=save_dir)
