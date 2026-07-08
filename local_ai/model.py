import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(max_len, dtype=torch.float)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_real, x_imag = x[..., :half], x[..., half:]
    seq_len = x.shape[2]
    cos = cos[:seq_len, :half].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :half].unsqueeze(0).unsqueeze(0)
    return torch.cat([
        x_real * cos - x_imag * sin,
        x_real * sin + x_imag * cos
    ], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        cos, sin = precompute_rope_freqs(self.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) + mask
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CustomTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm_final = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, loss_mask: torch.Tensor = None):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if loss_mask is not None:
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                mask_flat = loss_mask.view(-1).bool()
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], label_smoothing=0.1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95, eos_token_id: int = None, repetition_penalty: float = 1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                prev_tokens = idx[0].tolist()
                seen = set(prev_tokens)
                for token_id in seen:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.gather(1, sorted_indices.argsort(-1))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_model(config: ModelConfig = None):
    if config is None:
        config = ModelConfig()
    model = CustomTransformer(config)
    return model
