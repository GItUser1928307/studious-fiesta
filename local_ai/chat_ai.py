#!/usr/bin/env python3
"""
Better AI Chat - Uses more capable model
Works on Lenovo G570 (4GB RAM, Intel i3 CPU)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("  LOADING AI MODEL FROM HUGGINGFACE")
print("="*60)
print("\nThis will download a pre-trained model (~200-400MB)")
print("First time: 2-10 minutes depending on internet speed\n")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Optimize for your laptop
torch.set_num_threads(2)

# Try different models in order of preference
models_to_try = [
    # TinyStories-33M (story generator, works well)
    ("roneneldan/TinyStories-33M", "EleutherAI/gpt-neo-125M"),
    # GPT-2 small (124M, more general)
    ("gpt2", "gpt2"),
]

model_name, tokenizer_name = models_to_try[0]
print(f"Loading: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"Model loaded! {params:,} parameters ({params/1e6:.1f}M)")

print("\n" + "="*60)
print("  AI IS READY!")
print("="*60)
print("\nThis is a real pre-trained language model that:")
print("  - Understands English")
print("  - Generates coherent text")
print("  - Remembers context")
print("\nType your prompts below. Type 'quit' to exit.")
print("-"*60)

chat_context = ""

while True:
    try:
        prompt = input("\nYou: ").strip()
        
        if prompt.lower() == "quit":
            print("\nGoodbye!")
            break
        elif prompt.lower() == "clear":
            chat_context = ""
            print("Context cleared.")
            continue
        elif not prompt:
            continue
        
        # Add prompt to context
        if chat_context:
            full_prompt = chat_context + " " + prompt
        else:
            full_prompt = prompt
        
        print("AI is thinking...", end=" ", flush=True)
        
        # Tokenize
        inputs = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Get response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(full_prompt):].strip()
        
        if response_only:
            print(f"\nAI: {response_only}")
            # Keep context (last 200 chars)
            chat_context = (full_prompt + " " + response_only)[-300:]
        else:
            print("\nAI: (thinking...)")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Type 'quit' to exit.")
    except Exception as e:
        print(f"\nError: {e}")