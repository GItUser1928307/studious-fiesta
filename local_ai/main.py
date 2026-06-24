#!/usr/bin/env python3
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from tokenizer import list_tokenizers
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
        print("  info          - Show model architecture info")
        print("  generate      - Generate text from a trained model")
        print("  chat          - Interactive chat mode")
        print(f"  tokenizers    - List available tokenizers ({', '.join(list_tokenizers())})")
        print("\nOptions:")
        print("  --tokenizer NAME   Tokenizer to use (word, whitespace, char, gpt2)")
        print("  --model PATH       Path to model checkpoint")
        print("  --prompt TEXT      Text prompt for generation")
        print("\nExamples:")
        print("  python main.py info")
        print("  python main.py generate --prompt \"hello\"")
        print("  python main.py chat")
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

    elif cmd == "tokenizers":
        print("\nAvailable tokenizers:")
        for name in list_tokenizers():
            print(f"  {name}")
        print("\nUsage: python retrain.py --tokenizer <name>")
        print("       python main.py generate --tokenizer <name>")

    elif cmd == "generate":
        model_path = "quick_ckpt/best.pt"
        prompt = "hello how are you"
        tokenizer_name = None
        i = 1
        while i < len(args):
            if args[i] == "--model" and i + 1 < len(args):
                model_path = args[i + 1]
                i += 2
            elif args[i] == "--prompt" and i + 1 < len(args):
                prompt = args[i + 1]
                i += 2
            elif args[i] == "--tokenizer" and i + 1 < len(args):
                tokenizer_name = args[i + 1]
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
