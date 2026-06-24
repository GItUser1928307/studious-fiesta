#!/usr/bin/env python3
"""AI Chat - Local model only"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from system import print_system_info, get_cpu_threads
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
    print("No trained weights found - using random init")

model.eval()

params = model.count_params()
print(f"\nModel: {params:,} params ({params/1e6:.2f}M)")
print("\n" + "="*60)
print("  AI READY - Type prompts below. 'quit' to exit.")
print("="*60)

context = ""

while True:
    try:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            context = ""
            print("Context cleared.")
            continue
        if not prompt:
            continue

        full_prompt = context + " " + prompt if context else prompt
        ids = tokenizer.encode(full_prompt)
        idx = torch.tensor([ids]).long()

        print("Thinking...", end=" ", flush=True)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=80, temperature=0.8, top_k=40, top_p=0.95)

        new_tokens = out[0].tolist()[len(ids):]
        response = tokenizer.decode(new_tokens)

        if response.strip():
            print(f"\nAI: {response}")
            context = full_prompt + " " + response
            if len(tokenizer.encode(context)) > 500:
                context = full_prompt[-200:] + " " + response
        else:
            print("\nAI: [No response generated]")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
