#!/usr/bin/env python3
"""
AI MODEL - SETUP AND RUN
Single file to install dependencies, train (if needed), and chat.
Works on Windows with Python 3.8+
"""
import os
import sys
import subprocess
import importlib
import time

# ============================================================
# CONFIGURATION
# ============================================================
REQUIRED_PACKAGES = {
    "torch": "torch>=2.0.0",
    "tiktoken": "tiktoken>=0.5.0",
    "transformers": "transformers>=4.30.0",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
VENV_DIR = os.path.join(PARENT_DIR, "venv")
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "quick_ckpt")
TRAIN_DATA_FILE = os.path.join(PARENT_DIR, "quick_train_data.txt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final.pt")

# ============================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================
def check_and_install_packages():
    print("\n" + "=" * 60)
    print("  STEP 1: CHECKING DEPENDENCIES")
    print("=" * 60)

    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {module_name}")
        except ImportError:
            print(f"  [MISSING] {module_name}")
            missing.append(pip_name)

    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        for pkg in missing:
            print(f"  Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print("  All packages installed!")
    else:
        print("\n  All dependencies are installed.")

    # Verify imports work
    import torch
    import tiktoken
    import transformers
    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# STEP 2: CHECK FOR TRAINED MODEL
# ============================================================
def model_exists():
    return os.path.exists(BEST_MODEL_PATH) or os.path.exists(FINAL_MODEL_PATH)

def get_training_data():
    if os.path.exists(TRAIN_DATA_FILE):
        with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return None

# ============================================================
# STEP 3: TRAIN MODEL (if needed)
# ============================================================
def train_model():
    print("\n" + "=" * 60)
    print("  STEP 3: TRAINING AI MODEL")
    print("=" * 60)
    print("""
No trained model found. We need to train one first.

The training uses a short story about a girl named Lily and her cat.
Training takes about 1-3 minutes on CPU. The model will learn to
generate story-like text.

What the model learns:
  - Basic English words and sentence structure
  - Story patterns (characters, events, endings)
  - How to continue text from a prompt

""")
    input("Press ENTER to start training (or Ctrl+C to cancel)... ")

    import torch
    from model import create_model
    from tokenizer import CharTokenizer
    from config import SMALL_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = SMALL_CONFIG
    tokenizer = CharTokenizer()
    model = create_model(model_config).to(device)

    print(f"\n  Model parameters: {model.count_params():,}")
    print(f"  Architecture: {model_config.num_layers} layers, {model_config.hidden_size} hidden")
    print(f"  Training on: {device.upper()}")

    # Prepare training data if not exists
    if not os.path.exists(TRAIN_DATA_FILE):
        sample_text = """Once upon a time there was a little girl named Lily. She lived in a small house near a big forest. Every day she would play in the garden with her cat named Whiskers. Whiskers was a fluffy orange cat with big green eyes.

One sunny morning, Lily decided to explore the forest. She packed some cookies and juice in a small basket. Whiskers followed her as she walked into the trees.

Deep in the forest they found a sparkling stream. Fish of many colors swam in the clear water. Lily sat on a mossy rock and shared her cookies with Whiskers.

Suddenly they heard a strange noise. It came from behind a large bush. Lily was scared but curious. She slowly walked toward the bush and peeked through the leaves.

To her surprise she saw a tiny baby deer stuck in some vines. Its mother was nowhere to be seen. Lily carefully untangled the vines and freed the deer.

The baby deer licked her hand and ran off into the forest. Lily felt happy that she could help. She and Whiskers returned home just before sunset.

From that day on Lily visited the forest every weekend. She made many animal friends and learned that kindness makes the world a better place."""
        with open(TRAIN_DATA_FILE, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print("  Created training data file.")

    # Import training function
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn

    class TextDataset(Dataset):
        def __init__(self, file_path, tokenizer, seq_len):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.tokens = tokenizer.encode(text)
            self.seq_len = seq_len

        def __len__(self):
            return max(0, len(self.tokens) - self.seq_len)

        def __getitem__(self, idx):
            chunk = self.tokens[idx: idx + self.seq_len + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            return x, y

    dataset = TextDataset(TRAIN_DATA_FILE, tokenizer, model_config.max_seq_len)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model.train()
    total_steps = 0
    max_steps = 500
    start_time = time.time()

    print(f"\n  Training for {max_steps} steps...")
    print("  " + "-" * 50)

    while total_steps < max_steps:
        for x, y in loader:
            if total_steps >= max_steps:
                break

            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if total_steps % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {total_steps}/{max_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

            if total_steps % 200 == 0 and total_steps > 0:
                path = os.path.join(CHECKPOINT_DIR, f"step_{total_steps}.pt")
                torch.save({"model": model.state_dict(), "step": total_steps, "loss": loss.item()}, path)

            total_steps += 1

    # Save final model
    torch.save({"model": model.state_dict(), "config": model_config}, FINAL_MODEL_PATH)
    torch.save({"model": model.state_dict(), "config": model_config}, BEST_MODEL_PATH)

    elapsed = time.time() - start_time
    print("  " + "-" * 50)
    print(f"  Training complete! Time: {elapsed:.1f}s")
    print(f"  Model saved to: {CHECKPOINT_DIR}/")
    print(f"  Final loss: {loss.item():.4f}")

    return model, tokenizer

# ============================================================
# STEP 4: LOAD MODEL
# ============================================================
def load_model():
    print("\n" + "=" * 60)
    print("  STEP 4: LOADING AI MODEL")
    print("=" * 60)

    import torch
    from model import create_model
    from tokenizer import CharTokenizer
    from config import ModelConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find model file
    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        print("  ERROR: No trained model found!")
        return None, None, None

    print(f"  Loading model from: {os.path.basename(model_path)}")

    from config import SMALL_CONFIG

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config", None)

    if config_dict:
        if isinstance(config_dict, dict):
            config = ModelConfig(**config_dict)
        else:
            config = config_dict
    else:
        config = SMALL_CONFIG

    model = create_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    tokenizer = CharTokenizer()

    params = model.count_params()
    print(f"  Model loaded! {params:,} parameters ({params/1e6:.1f}M)")
    print(f"  Device: {device.upper()}")

    return model, tokenizer, device

# ============================================================
# STEP 5: INTERACTIVE CHAT
# ============================================================
def interactive_chat(model, tokenizer, device):
    print("\n" + "=" * 60)
    print("  AI MODEL READY!")
    print("=" * 60)
    print("""
This is a custom-trained language model that generates story text.

It was trained on a short story about Lily and her cat Whiskers.
The model learned English patterns, sentence structure, and story flow.

HOW TO USE:
  - Type any text and the model will continue it
  - Best results start with story-like prompts
  - Example: "Once upon a time" or "The cat went to"

COMMANDS:
  quit   - Exit the program
  clear  - Reset conversation history
  test   - Run sample generation tests
  help   - Show this help message
""")
    print("-" * 60)

    context = ""

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if prompt.lower() == "quit":
                print("\nGoodbye!")
                break
            elif prompt.lower() == "clear":
                context = ""
                print("Context cleared.")
                continue
            elif prompt.lower() == "help":
                print("\nCommands: quit, clear, test, help")
                print("Type any text to have the model continue it.")
                continue
            elif prompt.lower() == "test":
                run_tests(model, tokenizer, device)
                continue
            elif not prompt:
                continue

            # Build full prompt
            full_prompt = context + " " + prompt if context else prompt

            # Encode and generate
            input_ids = tokenizer.encode(full_prompt)
            idx = torch.tensor([input_ids], dtype=torch.long).to(device)

            print("Thinking...", end=" ", flush=True)
            with torch.no_grad():
                output = model.generate(idx, max_new_tokens=80, temperature=0.8, top_k=40, top_p=0.95)

            generated = output[0].tolist()
            new_tokens = generated[len(input_ids):]
            response = tokenizer.decode(new_tokens)

            if response.strip():
                print(f"\nAI: {response}")
                # Keep context for conversation flow
                context = full_prompt + " " + response
                # Trim context if too long
                if len(tokenizer.encode(context)) > 500:
                    context = full_prompt[-200:] + " " + response
            else:
                print("\nAI: [No response generated - try a different prompt]")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")

# ============================================================
# TEST FUNCTION
# ============================================================
def run_tests(model, tokenizer, device):
    print("\n" + "=" * 60)
    print("  RUNNING TESTS")
    print("=" * 60)

    test_prompts = [
        "Once upon a time",
        "The cat went to",
        "In a magical forest",
        "Lily found a",
        "The brave knight",
    ]

    for prompt in test_prompts:
        print(f"\n  Prompt: {prompt}")
        print("  " + "-" * 40)

        input_ids = tokenizer.encode(prompt)
        idx = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.95)

        generated = output[0].tolist()
        new_tokens = generated[len(input_ids):]
        result = tokenizer.decode(new_tokens)
        print(f"  Output: {prompt}{result}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETE!")
    print("=" * 60)

# ============================================================
# MAIN
# ============================================================
def main():
    # Check if running in venv, if not restart with venv python
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        if os.path.exists(venv_python):
            print("\n  Activating virtual environment...")
            subprocess.call([venv_python, os.path.abspath(__file__)] + sys.argv[1:])
            return

    print("\n" + "=" * 60)
    print("   AI MODEL - SETUP AND RUN")
    print("   Single file to install, train, and chat")
    print("=" * 60)

    # Step 1: Check dependencies
    try:
        check_and_install_packages()
    except Exception as e:
        print(f"\n  ERROR installing packages: {e}")
        print("  Try running: pip install torch tiktoken transformers")
        return

    # Step 2-3: Check for model / train
    if not model_exists():
        print("\n  No trained model found.")
        try:
            model, tokenizer = train_model()
            if model is None:
                return
        except KeyboardInterrupt:
            print("\n  Training cancelled.")
            return
        except Exception as e:
            print(f"\n  ERROR during training: {e}")
            return
    else:
        print("\n  Trained model found!")

    # Step 4: Load model
    model, tokenizer, device = load_model()
    if model is None:
        return

    # Step 5: Chat
    interactive_chat(model, tokenizer, device)

if __name__ == "__main__":
    main()
