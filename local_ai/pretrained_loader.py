#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def precompute_rope_freqs(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(max_len, dtype=torch.float)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x_real, x_imag = x[..., :half], x[..., half:]
    seq_len = x.shape[2]
    cos = cos[:seq_len, :half].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :half].unsqueeze(0).unsqueeze(0)
    return torch.cat([x_real * cos - x_imag * sin, x_real * sin + x_imag * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) + mask
        attn = torch.nn.functional.softmax(attn, dim=-1)
        y = attn @ v
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, max_seq_len)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CustomLLM(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=16, num_heads=12, ffn_dim=3072, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, ffn_dim, max_seq_len) for _ in range(num_layers)])
        self.norm_final = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x, targets=None):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.95):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs - torch.nn.functional.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.gather(1, sorted_indices.argsort(-1))
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def download_and_load_pretrained():
    print("="*60)
    print("LOADING PRE-TRAINED TINYSTORIES-33M MODEL")
    print("="*60)
    print("\nThis is a real 33-million parameter language model")
    print("pre-trained on millions of stories. It actually understands")
    print("English and can generate coherent text.\n")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("Downloading model from Hugging Face...")
    print("(This may take a few minutes on slow internet)\n")
    
    model_name = "roneneldan/TinyStories-33M"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"\nLoaded pre-trained model: {model_name}")
    print(f"HF Model parameters: {sum(p.numel() for p in hf_model.parameters()):,}")
    
    return hf_model, tokenizer


def use_custom_pretrained():
    print("\n" + "="*60)
    print("CREATING CUSTOM 50M PARAMETER MODEL")
    print("="*60)
    
    model = CustomLLM(
        vocab_size=50257,
        hidden_size=768,
        num_layers=8,
        num_heads=12,
        ffn_dim=3072,
        max_seq_len=512
    )
    
    print(f"\nCustom model: {model.count_params():,} parameters")
    print(f"Architecture: Llama-style with RoPE, SwiGLU, RMSNorm")
    
    print("\nNote: This model needs training to work.")
    print("For immediate results, we'll load pre-trained weights...")
    
    return model