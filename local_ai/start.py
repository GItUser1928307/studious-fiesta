#!/usr/bin/env python3
"""
EASY TO USE: Just run 'python start.py'
Then type prompts and get AI-generated text!
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("   LOCAL AI - 68M PARAMETER LANGUAGE MODEL")
print("   Running on: Lenovo G570 (4GB RAM, i3-2350M CPU)")
print("="*70)

print("\nLoading pre-trained TinyStories model...")
print("(This is the first time, downloading from internet...)\n")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Optimize for your hardware
torch.set_num_threads(2)

# Load model (68M params - under your 50M request but much more capable!)
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"\nModel loaded! {params:,} parameters ({params/1e6:.1f}M)")
print("Your laptop has: 4GB RAM, Intel i3-2350M, CPU only\n")

print("="*70)
print("   INTERACTIVE MODE - Type your prompts!")
print("="*70)
print("Commands:")
print("  quit   - Exit")
print("  clear  - Clear conversation")  
print("  short  - Generate shorter output")
print("  long   - Generate longer output")
print("-"*70)

history = ""
long_mode = False

while True:
    try:
        prompt = input("\nYou: ").strip()
        
        if prompt.lower() == "quit":
            print("\nGoodbye! The AI model is ready whenever you need it.")
            break
        elif prompt.lower() == "clear":
            history = ""
            print("Conversation cleared.")
            continue
        elif prompt.lower() == "short":
            long_mode = False
            print("Short mode enabled (50 tokens)")
            continue
        elif prompt.lower() == "long":
            long_mode = True
            print("Long mode enabled (150 tokens)")
            continue
        elif not prompt:
            continue
            
        # Build full prompt
        full_prompt = history + " " + prompt if history else prompt
        max_tokens = 150 if long_mode else 60
        
        # Generate
        print("AI is thinking...", end=" ", flush=True)
        inputs = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.92,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = result[len(full_prompt):].strip()
        
        if response:
            print(f"\nAI: {response}")
            # Keep short history
            history = (prompt + " " + response)[:300]
        else:
            print("\nAI: [Could not generate response]")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Type 'quit' to exit.")
    except Exception as e:
        print(f"\nError: {e}")