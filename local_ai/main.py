#!/usr/bin/env python3
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from tokenizer import WordTokenizer, CharTokenizer, GPT2Tokenizer
from train import train, train_quick, train_full
from generate import generate_text, interactive, load_model_and_tokenizer
from config import ModelConfig, SMALL_CONFIG


def count_params():
    print("Model parameter counts:")
    for name, cfg in [("Small (char-level)", SMALL_CONFIG), ("Full (BPE-level)", ModelConfig())]:
        model = create_model(cfg)
        params = model.count_params()
        print(f"  {name}: {params:,} params ({params/1e6:.1f}M)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} | Device: {device}")
    if device == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")

    args = sys.argv[1:] if len(sys.argv) > 1 else []

    if not args:
        print("\nUsage: python main.py <command> [options]")
        print("Commands:")
        print("  quick         - Quick demo: train a char-level model on sample text")
        print("  full          - Full training with BPE tokenizer (provide data.txt)")
        print("  generate      - Generate text from a trained model")
        print("  chat          - Interactive chat mode")
        print("  info          - Show model architecture info")
        print("\nExamples:")
        print("  python main.py quick")
        print("  python main.py generate --prompt \"Once upon a time\"")
        print("  python main.py chat")
        print("  python main.py info")
        return

    cmd = args[0]

    if cmd == "info":
        count_params()
        model = create_model()
        total = model.count_params()
        print(f"\nFull model architecture ({total:,} params / {total/1e6:.1f}M):")
        print(f"  Layers: {model.config.num_layers}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Heads: {model.config.num_heads} (head_dim={model.config.head_dim})")
        print(f"  FFN size: {model.config.intermediate_size}")
        print(f"  Vocab: {model.config.vocab_size}")
        print(f"  Max seq len: {model.config.max_seq_len}")
        print(f"  Weight tying: {model.config.tie_embeddings}")
        print(f"  Activation: SwiGLU + RoPE + RMSNorm")

    elif cmd == "quick":
        model, tokenizer = train_quick(device)
        print("\n--- Quick test ---")
        for prompt in ["Once upon", "Lily went", "The cat"]:
            out = generate_text(model, tokenizer, prompt, max_new=50, device=device)
            print(f"Prompt: {prompt!r}")
            print(f"Output: {out}\n")

    elif cmd == "full":
        if not os.path.exists("data.txt"):
            print("Error: data.txt not found. Create a text file with training data.")
            return
        model, tokenizer = train_full(device)
        print("\n--- Quick test ---")
        for prompt in ["The meaning"]:
            out = generate_text(model, tokenizer, prompt, max_new=50, device=device)
            print(f"Prompt: {prompt!r}")
            print(f"Output: {out}\n")

    elif cmd == "generate":
        model_path = "quick_ckpt/best.pt"
        prompt = "hello how are you"
        i = 1
        while i < len(args):
            if args[i] == "--model" and i + 1 < len(args):
                model_path = args[i + 1]
                i += 2
            elif args[i] == "--prompt" and i + 1 < len(args):
                prompt = args[i + 1]
                i += 2
            else:
                i += 1
        if not os.path.exists(model_path):
            print(f"Error: model not found at {model_path}")
            return
        model, tokenizer = load_model_and_tokenizer(model_path, device=device)
        out = generate_text(model, tokenizer, prompt, max_new=100, device=device)
        print(f"Prompt: {prompt!r}")
        print(f"Output: {out}")

    elif cmd == "chat":
        model_path = "quick_ckpt/best.pt"
        i = 1
        while i < len(args):
            if args[i] == "--model" and i + 1 < len(args):
                model_path = args[i + 1]
                i += 2
            else:
                i += 1
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Run 'python retrain.py' first.")
            return
        model, tokenizer = load_model_and_tokenizer(model_path, device=device)
        interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
