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

    # Normal default tokenizer: Byte-Level BPE 2048 (current pipeline)
    # Always build from current tokenizer.json or fresh BPE on train split
    if os.path.exists(TOKENIZER_PATH):
        try:
            tokenizer = load_tokenizer(TOKENIZER_PATH)
            # Prefer bytebpe; if it's old word 675, rebuild as bytebpe
            if tokenizer.vocab_size == 675 and os.path.exists(DATA_FILE):
                # Old word checkpoint still on disk — upgrade to default BPE
                print(f"Found old word tokenizer vocab 675, rebuilding default bytebpe")
                raise ValueError("upgrade to bytebpe")
        except Exception:
            # Build default bytebpe as retrain.py does (train split not needed for chat)
            from config import ModelConfig
            # Use current default: bytebpe 2048 (actual ~897 for tiny corpus)
            try:
                tokenizer = get_tokenizer("bytebpe", data_file=DATA_FILE, vocab_size=2048)
            except Exception:
                tokenizer = get_tokenizer("word", data_file=DATA_FILE)
    else:
        try:
            tokenizer = get_tokenizer("bytebpe", data_file=DATA_FILE, vocab_size=2048)
        except Exception:
            tokenizer = get_tokenizer("word", data_file=DATA_FILE)

    from config import ModelConfig
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_layers=22,
        num_heads=8,
        intermediate_size=1408,
        max_seq_len=96,
    )

    model = create_model(config)

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        ckpt_vocab = ckpt.get("config")
        if isinstance(ckpt_vocab, dict):
            ckpt_vocab = ckpt_vocab.get("vocab_size")
        elif hasattr(ckpt_vocab, "vocab_size"):
            ckpt_vocab = ckpt_vocab.vocab_size
        else:
            ckpt_vocab = None
        if ckpt_vocab is not None and ckpt_vocab != config.vocab_size:
            print(
                f"Checkpoint vocab {ckpt_vocab} != default tokenizer {config.vocab_size} "
                f"(old word 675 vs new BPE) — not loading old weights. "
                f"Train a new BPE checkpoint or delete {CKPT_PATH} to use current tokenizer."
            )
            print("Using random init with default bytebpe tokenizer")
        else:
            try:
                model.load_state_dict(ckpt["model"])
                print(f"Loaded trained weights! (vocab {config.vocab_size}, {model.count_params()/1e6:.1f}M)")
            except Exception as e:
                print(f"Failed to load checkpoint ({e}) — using random init with default tokenizer")
    else:
        print("No trained weights found - using random init with default bytebpe")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def ask_ai(model, tokenizer, device, prompt, context=""):
    full_prompt = f"{context} {prompt}".strip() if context else prompt

    # Tokenizer-agnostic: word vs Byte-BPE
    if hasattr(tokenizer, "word_to_id"):
        q_tokens = [t for t in tokenizer.tokenize(full_prompt) if t not in ("<q>", "<a>")]
        ids = [tokenizer.bos_token_id, tokenizer.q_token_id]
        ids += [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in q_tokens]
        ids.append(tokenizer.a_token_id)
    else:
        # Byte-Level BPE: encode raw text (byte fallback, no UNK)
        # Strip any <q>/<a> the user may have typed
        clean = full_prompt.replace("<q>", "").replace("<a>", "").strip()
        q_ids = tokenizer.encode(clean) if clean else []
        ids = [tokenizer.bos_token_id, tokenizer.q_token_id] + q_ids + [tokenizer.a_token_id]

    idx = torch.tensor([ids]).long().to(device)

    print("Thinking...", end=" ", flush=True)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=80,
            temperature=0.5,
            top_k=40,
            top_p=0.9,
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
