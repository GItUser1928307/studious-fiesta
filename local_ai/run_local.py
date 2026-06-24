#!/usr/bin/env python3
"""AI Chat - Local model only"""
import os, sys, gc
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
print("  AI READY")
print("="*70)
print("Commands: quit, clear, test, help\n")

history = ""

while True:
    try:
        prompt = input("You: ").strip()
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            history = ""
            print("Cleared.")
            continue
        if prompt.lower() == "test":
            for p in ["Once upon a time", "The cat went to", "Hello there"]:
                ids = tokenizer.encode(p)
                idx = torch.tensor([ids]).long()
                with torch.no_grad():
                    out = model.generate(idx, max_new_tokens=50)
                print(f"\n>>> {p}{tokenizer.decode(out[0].tolist()[len(ids):])}")
            continue
        if prompt.lower() == "help":
            print("Commands: quit, clear, test, help")
            continue
        if not prompt:
            continue

        full_prompt = history + " " + prompt if history else prompt
        ids = tokenizer.encode(full_prompt)
        idx = torch.tensor([ids]).long()

        print("Thinking...")
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.92)

        new_tokens = out[0].tolist()[len(ids):]
        response = tokenizer.decode(new_tokens)

        if response.strip():
            print(f"\nAI: {response}")
            history = full_prompt + " " + response
            if len(tokenizer.encode(history)) > 500:
                history = full_prompt[-200:] + " " + response
        else:
            print("\nAI: [No response generated]")

        gc.collect()

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

del model
gc.collect()
