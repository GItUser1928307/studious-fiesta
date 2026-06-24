#!/usr/bin/env python3
"""AI Chat - Local model only"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from system import print_system_info, get_cpu_threads
from model import create_model
from tokenizer import load_tokenizer, get_tokenizer
from config import auto_config_from_data
from generate import clean_response

info = print_system_info()
torch.set_num_threads(info["threads"])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CKPT_PATH = os.path.join(PARENT_DIR, "quick_ckpt", "best.pt")
TOKENIZER_PATH = os.path.join(PARENT_DIR, "quick_ckpt", "tokenizer.json")
DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")

if os.path.exists(TOKENIZER_PATH):
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    config = auto_config_from_data(DATA_FILE)
elif os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    tokenizer_name = ckpt.get("tokenizer", "word")
    tokenizer = get_tokenizer(tokenizer_name, data_file=DATA_FILE)
    config = auto_config_from_data(DATA_FILE)
else:
    tokenizer = get_tokenizer("word", data_file=DATA_FILE)
    config = auto_config_from_data(DATA_FILE)

model = create_model(config)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("Loaded trained weights!")
else:
    print("No trained weights found - using random init")

model.eval()

params = model.count_params()
print(f"\nModel: {params:,} params ({params/1e6:.2f}M)")
print(f"Tokenizer: {tokenizer.vocab_size} vocab")
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

        q_tokens = [t for t in tokenizer.tokenize(full_prompt) if t not in ("<q>", "<a>")]
        ids = [tokenizer.bos_token_id, tokenizer.q_token_id]
        ids += [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in q_tokens]
        ids.append(tokenizer.a_token_id)

        idx = torch.tensor([ids]).long()

        print("Thinking...", end=" ", flush=True)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=80, temperature=0.3, top_k=20, top_p=0.8, eos_token_id=tokenizer.eos_token_id)

        new_tokens = out[0].tolist()[len(ids):]
        response = clean_response(tokenizer.decode(new_tokens))

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
