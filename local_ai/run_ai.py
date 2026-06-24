#!/usr/bin/env python3
import os
import sys
import torch

def main():
    print("\n" + "="*70)
    print("           LOCAL AI MODEL - 50M PARAMETER TRANSFORMER")
    print("="*70)
    print(f"\nPyTorch: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    print(f"CPU Threads: {torch.get_num_threads()}")
    
    print("\n" + "-"*70)
    print("OPTIONS:")
    print("-"*70)
    print("  1. Load PRE-TRAINED TinyStories-33M (works immediately!)")
    print("  2. Load PRE-TRAINED TinyStories-8M (faster, smaller)")  
    print("  3. Train from scratch (takes time, needs good data)")
    print("  4. Test custom model (untrained)")
    print("  5. Interactive chat with pre-trained model")
    print("-"*70)
    
    choice = input("\nChoose (1-5): ").strip()
    
    if choice == "1":
        load_tinystories_33m()
    elif choice == "2":
        load_tinystories_8m()
    elif choice == "3":
        train_from_scratch()
    elif choice == "4":
        test_custom()
    elif choice == "5":
        chat_mode()
    else:
        print("Invalid choice")


def load_tinystories_33m():
    print("\n" + "="*70)
    print("LOADING TINYSTORIES-33M - A REAL PRE-TRAINED AI MODEL")
    print("="*70)
    print("\nTinyStories-33M is a 33 million parameter transformer trained")
    print("on millions of short stories. It understands English and generates")
    print("coherent, meaningful text.\n")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Downloading model from Hugging Face...")
        print("(First time may take 5-15 minutes depending on internet)\n")
        
        model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        
        print(f"\nModel loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("\n" + "-"*70)
        print("GENERATION TEST")
        print("-"*70)
        
        prompts = [
            "Once upon a time there was a brave little rabbit",
            "The princess went to the magical forest and found",
            "In a land far away there lived a wise old owl"
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 50)
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(inputs, max_new_tokens=80, temperature=0.7, 
                                   top_p=0.92, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(result)
        
        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE MODE - Type your prompts!")
        print("="*70)
        print("Type 'quit' to exit, 'clear' to reset\n")
        
        history = ""
        while True:
            try:
                prompt = input("You: ")
                if prompt.lower() == "quit":
                    break
                if prompt.lower() == "clear":
                    history = ""
                    print("Conversation cleared.")
                    continue
                
                full_prompt = history + " " + prompt if history else prompt
                inputs = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=500)
                outputs = model.generate(inputs, max_new_tokens=100, temperature=0.7,
                                       top_p=0.92, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                print(f"AI: {response}\n")
                history = full_prompt + " " + response
                if len(history) > 1000:
                    history = ""
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have internet connection for first download.")


def load_tinystories_8m():
    print("\n" + "="*70)
    print("LOADING TINY-LM-8M - A SMALLER PRE-TRAINED MODEL")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("\nDownloading TinyLM-8M model...")
        model = AutoModelForCausalLM.from_pretrained("sixf0ur/tiny-lm-8M")
        tokenizer = AutoTokenizer.from_pretrained("sixf0ur/tiny-lm-8M")
        
        print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        prompts = ["The meaning of life is", "Once upon a time in a magical kingdom"]
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(inputs, max_new_tokens=60, temperature=0.7, do_sample=True)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            
    except Exception as e:
        print(f"Error: {e}")


def train_from_scratch():
    print("\n" + "="*70)
    print("TRAINING FROM SCRATCH")
    print("="*70)
    print("\nThis will train a custom transformer on your data.")
    print("Training takes time on CPU, but you'll have a unique model.")
    
    data_file = input("\nEnter training data file path: ").strip()
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    # Import and run training
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import train_full
    train_full("cpu")


def test_custom():
    print("\n" + "="*70)
    print("TESTING CUSTOM UNTRAINED MODEL")
    print("="*70)
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import create_model
    from tokenizer import CharTokenizer
    import torch
    
    print("\nCreating character-level model...")
    from config import SMALL_CONFIG
    model = create_model(SMALL_CONFIG)
    tokenizer = CharTokenizer()
    
    print(f"Model parameters: {model.count_params():,}")
    
    prompts = ["Once upon", "The cat", "Hello"]
    print("\nUntrained outputs (random):")
    for p in prompts:
        ids = tokenizer.encode(p)
        idx = torch.tensor([ids]).long()
        out = model.generate(idx, max_new_tokens=30)
        print(f"\nPrompt: {p}")
        print(f"Output: {tokenizer.decode(out[0].tolist())}")


def chat_mode():
    load_tinystories_33m()


if __name__ == "__main__":
    main()