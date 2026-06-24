#!/usr/bin/env python3
"""
Local AI Model for Lenovo G570 (4GB RAM, i3-2350M CPU, Intel HD 3000)
Optimized for low-memory CPU-only inference
"""
import os
import gc
import torch
import sys

print("\n" + "="*70)
print("    LOCAL AI MODEL - OPTIMIZED FOR YOUR LAPTOP")
print("="*70)

# Check hardware
print("\nHardware Info:")
print(f"  CPU: Intel i3-2350M @ 2.30GHz (2 cores, 4 threads)")
print(f"  RAM: 4GB DDR3")
print(f"  GPU: Intel HD 3000 (no CUDA)")
print(f"  PyTorch: {torch.__version__}")
print(f"  Device: CPU (no GPU acceleration)")

# Memory optimization
torch.set_num_threads(2)  # Use 2 threads for CPU

def load_model_optimized():
    """Load TinyStories-33M optimized for 4GB RAM"""
    print("\n" + "="*70)
    print("LOADING PRE-TRAINED TINYSTORIES MODEL")
    print("="*70)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\nModel: TinyStories-33M (68.5M parameters)")
    print("Loading from Hugging Face...")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    print("Tokenizer loaded")
    
    # Load model with memory optimization
    print("Loading model (this needs ~2GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    
    # Set to evaluation mode
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded! Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=80, temp=0.7):
    """Generate text with memory-efficient settings"""
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def interactive_chat():
    """Interactive chat mode optimized for low memory"""
    model, tokenizer = load_model_optimized()
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Type your prompts below. The model will generate text.")
    print("Commands:")
    print("  quit - Exit the program")
    print("  clear - Clear conversation history")
    print("  test  - Run a quick generation test")
    print("-"*70)
    
    history = ""
    test_count = 0
    
    while True:
        try:
            print()
            prompt = input("You: ").strip()
            
            if prompt.lower() == "quit":
                print("\nGoodbye! Thanks for using the AI model.")
                break
            elif prompt.lower() == "clear":
                history = ""
                print("Conversation cleared.")
                continue
            elif prompt.lower() == "test":
                print("\nRunning test generation...")
                test_prompts = [
                    "Once upon a time there was",
                    "The little cat went to",
                    "In a magical kingdom"
                ]
                for p in test_prompts:
                    result = generate_text(model, tokenizer, p, max_tokens=50)
                    print(f"\nPrompt: {p}")
                    print(f"Output: {result}")
                continue
            elif prompt.lower() == "help":
                print("Commands: quit, clear, test, help")
                continue
            elif not prompt:
                continue
            
            # Build full prompt from history
            full_prompt = history + " " + prompt if history else prompt
            
            # Generate
            print("Thinking...")
            result = generate_text(model, tokenizer, full_prompt, max_tokens=100)
            
            # Extract only the new generated part
            new_text = result[len(full_prompt):].strip()
            
            if new_text:
                print(f"\nAI: {new_text}")
                history = result[:500]  # Keep limited history
            else:
                print("\nAI: [No response generated]")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit properly.")
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    del model
    gc.collect()


def quick_test():
    """Run a quick test to verify the model works"""
    model, tokenizer = load_model_optimized()
    
    print("\n" + "="*70)
    print("QUICK GENERATION TEST")
    print("="*70)
    
    test_prompts = [
        "Once upon a time there was a brave little rabbit",
        "The princess found a magical key and",
        "In a land far away there lived a wise"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/3 ---")
        print(f"Prompt: {prompt}")
        result = generate_text(model, tokenizer, prompt, max_tokens=60)
        print(f"Output: {result}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    
    del model
    gc.collect()


if __name__ == "__main__":
    print("\nChoose an option:")
    print("  1. Interactive Chat (talk to the AI)")
    print("  2. Quick Test (run 3 test generations)")
    print("  3. Just load and show model info")
    
    choice = input("\nEnter (1/2/3): ").strip()
    
    if choice == "1":
        interactive_chat()
    elif choice == "2":
        quick_test()
    elif choice == "3":
        model, tokenizer = load_model_optimized()
        print(f"\nModel ready in memory.")
    else:
        print("Invalid choice, running quick test...")
        quick_test()