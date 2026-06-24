#!/usr/bin/env python3
"""Quick test - local model only"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from system import print_system_info
from model import create_model
from tokenizer import CharTokenizer
from config import auto_config

info = print_system_info()
torch.set_num_threads(info["threads"])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "quick_ckpt", "best.pt")

config = auto_config()
model = create_model(config)
tokenizer = CharTokenizer()

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("Loaded trained weights!")
else:
    print("No trained weights - using random init")

model.eval()
print(f"Model: {model.count_params():,} params ({model.count_params()/1e6:.2f}M)")

print("\n" + "-"*60)
print("GENERATING TEXT")
print("-"*60)

for prompt in ["Once upon a time", "The cat went to", "Hello there"]:
    print(f"\n>>> {prompt}")
    print("-" * 40)
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids]).long()
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=60, temperature=0.7, top_k=40, top_p=0.92)
    print(tokenizer.decode(out[0].tolist()))

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
