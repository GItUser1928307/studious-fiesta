#!/usr/bin/env python3
"""AI Chat - Local model only"""
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
    print("No trained weights found - using random init")

model.eval()

print("\n" + "="*70)
print("   AI READY - Type prompts below!")
print("="*70)
print("Commands: quit, clear, short, long\n")

history = ""
long_mode = False

while True:
    try:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            history = ""
            print("Conversation cleared.")
            continue
        if prompt.lower() == "short":
            long_mode = False
            print("Short mode (50 tokens)")
            continue
        if prompt.lower() == "long":
            long_mode = True
            print("Long mode (150 tokens)")
            continue
        if not prompt:
            continue

        full_prompt = history + " " + prompt if history else prompt
        max_new = 150 if long_mode else 60

        print("Thinking...", end=" ", flush=True)
        ids = tokenizer.encode(full_prompt)
        idx = torch.tensor([ids]).long()

        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=max_new, temperature=0.7, top_k=40, top_p=0.92)

        new_tokens = out[0].tolist()[len(ids):]
        response = tokenizer.decode(new_tokens)

        if response.strip():
            print(f"\nAI: {response}")
            history = (prompt + " " + response)[:300]
        else:
            print("\nAI: [No response generated]")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
