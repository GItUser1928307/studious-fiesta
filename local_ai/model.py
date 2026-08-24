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
        # rsqrt is generally a good XLA-friendly formulation of RMSNorm.
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def precompute_rope_freqs(
    dim: int,
    max_len: int,
    theta: float = 10000.0,
):
    """
    Precompute rotary embedding frequencies on CPU.

    These buffers are registered on the model and automatically move
    to the accelerator when model.to(device) is called.
    """
    freqs = 1.0 / (
        theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
    )

    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings.

    x shape:
        [batch, heads, sequence, head_dim]

    cos/sin shape:
        [sequence, head_dim // 2]
    """
    half = x.shape[-1] // 2

    x_real = x[..., :half]
    x_imag = x[..., half:]

    seq_len = x.shape[2]

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    rotated_real = x_real * cos - x_imag * sin
    rotated_imag = x_real * sin + x_imag * cos

    return torch.cat(
        [rotated_real, rotated_imag],
        dim=-1,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.hidden_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.qkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False,
        )

        self.out = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        cos, sin = precompute_rope_freqs(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin,
        )

        # Static sequence length is useful for XLA because it helps
        # prevent unnecessary recompilations.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        attention_scores = attention_scores * (
            self.head_dim ** -0.5
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        y = torch.matmul(
            attention_weights,
            v,
        )

        y = y.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            hidden_size,
        )

        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
    ):
        super().__init__()

        self.gate = nn.Linear(
            d_model,
            ffn_dim,
            bias=False,
        )

        self.up = nn.Linear(
            d_model,
            ffn_dim,
            bias=False,
        )

        self.down = nn.Linear(
            ffn_dim,
            d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate(x))
        up = self.up(x)

        return self.down(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm1 = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.attn = CausalSelfAttention(config)

        self.norm2 = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.ffn = SwiGLU(
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


class CustomTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.norm_final = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(
                module,
                (nn.Linear, nn.Embedding),
            ):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
    ):
        x = self.token_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_final(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            logits_flat = logits.reshape(
                -1,
                logits.size(-1),
            )

            targets_flat = targets.reshape(-1)

            if loss_mask is not None:
                mask_flat = loss_mask.reshape(-1).bool()

                selected_logits = logits_flat[mask_flat]
                selected_targets = targets_flat[mask_flat]

                loss = F.cross_entropy(
                    selected_logits,
                    selected_targets,
                    label_smoothing=0.1,
                )
            else:
                loss = F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    label_smoothing=0.1,
                )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_token_id: int = None,
        repetition_penalty: float = 1.0,
    ):
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
                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                logits = logits.masked_fill(
                    logits < values[:, -1:],
                    float("-inf"),
                )

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                )

                sorted_probs = F.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1,
                )

                sorted_mask = cumulative_probs > top_p

                sorted_mask[:, 1:] = sorted_mask[
                    :, :-1
                ].clone()

                sorted_mask[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(
                    sorted_mask,
                    float("-inf"),
                )

                logits = sorted_logits.gather(
                    1,
                    sorted_indices.argsort(dim=-1),
                )

            probs = F.softmax(
                logits,
                dim=-1,
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                [idx, idx_next],
                dim=1,
            )

            if (
                eos_token_id is not None
                and idx_next.item() == eos_token_id
            ):
                break

        return idx

    def count_params(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )


def create_model(config: ModelConfig = None):
    if config is None:
        config = ModelConfig()

    return CustomTransformer(config)
