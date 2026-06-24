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

print("\n" + "="*60)
print("  AI READY - Type prompts below. 'quit' to exit.")
print("="*60)

history = []

while True:
    try:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            history = []
            print("Context cleared.")
            continue
        if not prompt:
            continue

        context = " ".join(history[-3:])
        full_prompt = context + " " + prompt if context else prompt

        print("Thinking...", end=" ", flush=True)
        ids = tokenizer.encode(full_prompt)
        idx = torch.tensor([ids]).long()

        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.92)

        new_tokens = out[0].tolist()[len(ids):]
        response = tokenizer.decode(new_tokens)

        if response.strip():
            print(f"\nAI: {response}")
            history.append(prompt)
            history.append(response)
        else:
            print("\nAI: [No response generated]")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
