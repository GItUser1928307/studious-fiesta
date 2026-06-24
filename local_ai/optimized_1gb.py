#!/usr/bin/env python3
"""
AI Model - Optimized for 1GB RAM
Works on Lenovo G570 with limited resources
"""
import os
import gc
import torch

# Memory optimization - USE LESS THAN 1GB RAM
# Set pytorch to use minimal memory

# 1. Disable gradient computation globally
torch.set_grad_enabled(False)

# 2. Use only 2 threads for CPU
torch.set_num_threads(2)

# 3. Set memory fraction to prevent caching
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5)

print("="*60)
print("OPTIMIZED AI MODEL - 1GB RAM EDITION")
print("="*60)
print(f"\nDevice: CPU")
print(f"Threads: {torch.get_num_threads()}")
print(f"Memory optimization: Enabled")

# Now load model
print("\nLoading model with memory optimization...")

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Try to load with quantization for lower memory
# This reduces model size significantly

print("\nDownloading and loading TinyStories-33M model...")

try:
    # First try normal load
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    
    # Apply dynamic quantization to reduce memory
    print("\nApplying memory optimization (dynamic quantization)...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
except Exception as e:
    print(f"Quantization failed: {e}")
    print("Loading with standard method...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")

model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"\nModel loaded! {params:,} parameters ({params/1e6:.1f}M)")

# Check memory usage
import psutil
process = psutil.Process()
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {mem_mb:.0f} MB ({mem_mb/1024:.2f} GB)")

print("\n" + "="*60)
print("AI MODEL READY!")
print("="*60)

# Generation function with memory management
def generate(prompt, max_tokens=60):
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive mode
print("\nType prompts below. Type 'quit' to exit.")
print("-"*60)

while True:
    try:
        prompt = input("\nYou: ").strip()
        
        if prompt.lower() == "quit":
            print("\nGoodbye!")
            break
        elif not prompt:
            continue
            
        print("Thinking...", end=" ", flush=True)
        result = generate(prompt)
        print(f"\nAI: {result}")
        
        # Clear cache periodically
        gc.collect()
        
    except KeyboardInterrupt:
        print("\nInterrupted. Type 'quit' to exit.")
    except Exception as e:
        print(f"Error: {e}")