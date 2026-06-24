#!/usr/bin/env python3
"""Quick test - run this to see the AI working!"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("QUICK TEST - Generating AI Text")
print("="*60)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.set_num_threads(2)

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

print("\n" + "-"*60)
print("GENERATING TEXT")
print("-"*60)

test_prompts = [
    "Once upon a time there was a little dragon who",
    "The magic forest held many secrets, and one day",
    "A brave knight went on an adventure to find"
]

for prompt in test_prompts:
    print(f"\n>>> {prompt}")
    print("-" * 40)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

print("\n" + "="*60)
print("TEST COMPLETE! Your AI model is working!")
print("="*60)
print("\nTo chat interactively, run: python start.py")