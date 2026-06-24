#!/usr/bin/env python3
"""
Simple AI Chat - Just run this file!
Works on your Lenovo G570 (4GB RAM, i3 CPU)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("  LOADING AI MODEL FROM HUGGINGFACE")
print("  Your Laptop: 4GB RAM, Intel i3-2350M CPU")
print("="*60)

# Load the pre-trained model
from transformers import pipeline
import torch

# Use CPU only, optimize for low memory
torch.set_num_threads(2)

print("\nDownloading and loading model...")
print("(First time: takes 2-5 minutes)\n")

# TinyStories-33M - a real pre-trained language model
# It understands English and generates coherent text
pipe = pipeline(
    "text-generation", 
    model="roneneldan/TinyStories-33M",
    tokenizer="EleutherAI/gpt-neo-125M",
    device=-1  # CPU
)

print("="*60)
print("  AI MODEL READY!")
print("="*60)
print("\nThe model has been trained on millions of stories.")
print("It understands English and produces coherent responses.\n")
print("Commands:")
print("  quit   - Exit")
print("  clear - Clear chat history")
print("-"*60)

chat_history = []

while True:
    try:
        prompt = input("\nYou: ").strip()
        
        if prompt.lower() == "quit":
            print("\nGoodbye!")
            break
        elif prompt.lower() == "clear":
            chat_history = []
            print("Chat cleared.")
            continue
        elif not prompt:
            continue
        
        # Build context from history
        context = " ".join(chat_history[-3:])  # Last 3 messages
        full_prompt = context + " " + prompt if context else prompt
        
        print("AI is thinking...", end=" ", flush=True)
        
        # Generate response
        result = pipe(
            full_prompt,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.92,
            do_sample=True,
            pad_token_id=50256
        )
        
        response = result[0]["generated_text"][len(full_prompt):].strip()
        
        if response:
            print(f"\nAI: {response}")
            chat_history.append(prompt)
            chat_history.append(response)
        else:
            print("\nAI: (thinking...)")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Type 'quit' to exit.")
    except Exception as e:
        print(f"\nError: {e}")