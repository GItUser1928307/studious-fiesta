#!/usr/bin/env python3
"""AI Chat - Local model only"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from system import print_system_info
from model import create_model
from tokenizer import load_tokenizer, get_tokenizer
from config import auto_config_from_data
from generate import clean_response


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CKPT_PATH = os.path.join(PARENT_DIR, "quick_ckpt", "best.pt")
TOKENIZER_PATH = os.path.join(PARENT_DIR, "quick_ckpt", "tokenizer.json")
DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")


def load_everything():
    info = print_system_info()
    torch.set_num_threads(info["threads"])

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def ask_ai(model, tokenizer, device, prompt, context=""):
    full_prompt = f"{context} {prompt}".strip() if context else prompt

    q_tokens = [t for t in tokenizer.tokenize(full_prompt) if t not in ("<q>", "<a>")]
    ids = [tokenizer.bos_token_id, tokenizer.q_token_id]
    ids += [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in q_tokens]
    ids.append(tokenizer.a_token_id)

    idx = torch.tensor([ids]).long().to(device)

    print("Thinking...", end=" ", flush=True)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=80,
            temperature=0.3,
            top_k=20,
            top_p=0.8,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0].tolist()[len(ids):]
    response = clean_response(tokenizer.decode(new_tokens)).strip()

    if response:
        new_context = full_prompt + " " + response
        if len(tokenizer.encode(new_context)) > 500:
            new_context = full_prompt[-200:] + " " + response
        return response, new_context

    return "[No response generated]", context


def main():
    parser = argparse.ArgumentParser(description="One-shot AI chat test for Kaggle/terminal.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="hi",
        help="Message to send to the AI. Default: hi",
    )
    args = parser.parse_args()

    model, tokenizer, device = load_everything()

    params = model.count_params()
    print(f"Device: {device.upper()}")
    print(f"\nModel: {params:,} params ({params/1e6:.2f}M)")
    print(f"Tokenizer: {tokenizer.vocab_size} vocab")
    print("\n" + "=" * 60)
    print("  AI READY - Sending one test prompt.")
    print("=" * 60)

    context = ""
    prompt = args.prompt.strip() or "hi"

    print(f"\nYou: {prompt}")
    try:
        response, context = ask_ai(model, tokenizer, device, prompt, context)
        print(f"\nAI: {response}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return


if __name__ == "__main__":
    main()
