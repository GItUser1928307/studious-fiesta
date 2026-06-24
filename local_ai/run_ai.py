#!/usr/bin/env python3
"""AI Menu - Local model only"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from system import print_system_info
from model import create_model
from tokenizer import CharTokenizer
from config import auto_config, SMALL_CONFIG

info = print_system_info()
torch.set_num_threads(info["threads"])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "quick_ckpt", "best.pt")


def load_model():
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
    return model, tokenizer


def chat(model, tokenizer):
    print("\n" + "="*70)
    print("  INTERACTIVE MODE - Type prompts! 'quit' to exit.")
    print("="*70)
    history = ""
    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() == "quit":
                break
            if prompt.lower() == "clear":
                history = ""
                print("Cleared.")
                continue
            if not prompt:
                continue

            full_prompt = history + " " + prompt if history else prompt
            ids = tokenizer.encode(full_prompt)
            idx = torch.tensor([ids]).long()

            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.92)

            new_tokens = out[0].tolist()[len(ids):]
            response = tokenizer.decode(new_tokens)

            if response.strip():
                print(f"AI: {response}\n")
                history = full_prompt + " " + response
                if len(history) > 1000:
                    history = ""
            else:
                print("AI: [No response generated]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def test_untrained():
    print("\n" + "="*70)
    print("  UNTRAINED MODEL TEST")
    print("="*70)
    model = create_model(SMALL_CONFIG)
    tokenizer = CharTokenizer()
    print(f"Params: {model.count_params():,}")
    for p in ["Once upon", "The cat", "Hello"]:
        ids = tokenizer.encode(p)
        idx = torch.tensor([ids]).long()
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=30)
        print(f"\n>>> {p}")
        print(tokenizer.decode(out[0].tolist()))


def main():
    print("\n" + "="*70)
    print("           LOCAL AI MODEL")
    print("="*70)
    print(f"\nPyTorch: {torch.__version__}")
    print(f"Device:  {info['device'].upper()}")
    print(f"CPU:     {info['cpu']}")
    print(f"RAM:     {info['avail_ram_gb']:.1f}GB available")
    print(f"Threads: {info['threads']}")
    print("\n" + "-"*70)
    print("OPTIONS:")
    print("-"*70)
    print("  1. Chat with trained model")
    print("  2. Test untrained model")
    print("  3. Show model info")
    print("-"*70)

    choice = input("\nChoose (1-3): ").strip()

    if choice == "1":
        model, tokenizer = load_model()
        chat(model, tokenizer)
    elif choice == "2":
        test_untrained()
    elif choice == "3":
        model, _ = load_model()
        params = model.count_params()
        print(f"\nParams: {params:,} ({params/1e6:.2f}M)")
        print(f"Layers: {model.config.num_layers}")
        print(f"Hidden: {model.config.hidden_size}")
        print(f"Heads:  {model.config.num_heads}")
        print(f"FFN:    {model.config.intermediate_size}")
        print(f"Seq:    {model.config.max_seq_len}")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
