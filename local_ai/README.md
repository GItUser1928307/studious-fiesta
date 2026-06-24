# Local AI Model

## Quick Start (Recommended)

Just run one file - it handles everything:

```bash
cd D:\Omar\AI_MODEL\local_ai
python setup_and_run.py
```

This will:
1. Install any missing packages (torch, tiktoken, transformers)
2. Find your trained model (or train a new one if needed)
3. Start an interactive chat

## What's In This Project

### Files
- `setup_and_run.py` - **Run this!** Single file to install, setup, and chat
- `model.py` - Custom transformer architecture (RoPE, SwiGLU, RMSNorm)
- `config.py` - Model configuration
- `tokenizer.py` - Tokenizers (GPT-2 BPE and character-level)
- `train.py` - Training code
- `generate.py` - Text generation functions
- `main.py` - CLI with multiple commands

### Other Runners (require HuggingFace transformers)
- `chat_ai.py` - Uses TinyStories-33M from HuggingFace
- `start.py` - Another HuggingFace-based runner
- `test.py` - Quick test with HuggingFace model

## Your Model

You have a **trained custom model** in `D:\Omar\AI_MODEL\quick_ckpt/`:
- `best.pt` - Best checkpoint (25MB)
- `final.pt` - Final checkpoint
- Trained on a story about Lily and her cat Whiskers
- ~3.7M parameters, character-level tokenizer

## Manual Usage

### Chat with your trained model
```bash
python setup_and_run.py
```

### Use main.py commands
```bash
python main.py info      # Show model architecture
python main.py generate  # Generate text from trained model
python main.py chat      # Interactive chat
```

## Model Details

- **Architecture**: Custom Transformer with RoPE, SwiGLU, RMSNorm
- **Parameters**: ~3.7M (small, fast, fits in 4GB RAM)
- **Tokenizer**: Character-level
- **Training**: 500 steps on short story text

## Hardware

- **Laptop**: Lenovo G570
- **CPU**: Intel i3-2350M @ 2.30GHz
- **RAM**: 4GB DDR3
- **GPU**: Intel HD 3000 (CPU-only)

---
Made with PyTorch