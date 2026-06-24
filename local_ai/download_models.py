#!/usr/bin/env python3
"""
Download AI models from HuggingFace to local directory
Optimized for 1GB RAM on Lenovo G570
"""
import os
import torch

print("="*60)
print("DOWNLOADING AI MODELS FROM HUGGINGFACE")
print("="*60)

# Create models directory
os.makedirs("models", exist_ok=True)
os.makedirs("models_cache", exist_ok=True)

# Set cache directory
os.environ["HF_HOME"] = os.path.abspath("models_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("models_cache")

print("\nDownloading models...")
print("(First time may take 10-30 minutes)\n")

# We'll download a small but capable model
# Using GPT-2 small which can work with 1GB RAM when optimized

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

# Option 1: GPT-2 Small (124M params, ~500MB)
print("="*60)
print("OPTION 1: Downloading GPT-2 Small (124M params)")
print("="*60)

print("\nDownloading tokenizer...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.save_pretrained("models/gpt2-tokenizer")

print("Downloading model...")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.save_pretrained("models/gpt2-model")

print("GPT-2 saved to models/gpt2-model/")

# Option 2: TinyStories-33M (more coherent but larger)
print("\n" + "="*60)
print("OPTION 2: Downloading TinyStories-33M (68M params)")  
print("="*60)

print("\nDownloading TinyStories model...")
tinystories_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tinystories_tokenizer.save_pretrained("models/tinystories-tokenizer")

tinystories_model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
tinystories_model.save_pretrained("models/tinystories-model")

print("TinyStories saved to models/tinystories-model/")

# Check file sizes
print("\n" + "="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)

import glob
total_size = 0
for f in glob.glob("models/**/*", recursive=True):
    if os.path.isfile(f):
        size = os.path.getsize(f)
        total_size += size
        print(f"  {f}: {size/1e6:.1f} MB")

print(f"\nTotal: {total_size/1e6:.1f} MB")

print("\nNow creating optimized 1GB RAM version...")

# Create optimized version with quantization
# This creates a smaller model that fits in 1GB RAM
print("\nCreating 1GB RAM optimized model...")