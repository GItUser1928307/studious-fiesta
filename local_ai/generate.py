import torch
from model import create_model
from tokenizer import GPT2Tokenizer, CharTokenizer
from config import ModelConfig


def load_model_and_tokenizer(model_path: str, use_char: bool = False, device: str = "cpu"):
    from config import SMALL_CONFIG
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config", None)
    if config_dict:
        config = ModelConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
    else:
        config = SMALL_CONFIG
    model = create_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    tokenizer = CharTokenizer() if use_char else GPT2Tokenizer()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new: int = 100, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95, device: str = "cpu"):
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long).to(device)
    output = model.generate(idx, max_new, temperature, top_k, top_p)
    generated = output[0].tolist()
    new_tokens = generated[len(input_ids):]
    return tokenizer.decode(new_tokens)


def interactive(model, tokenizer, device: str):
    print("\n=== Interactive Generation ===")
    print("Type 'quit' to exit, 'reset' to clear context")
    print("Press Ctrl+C to stop generation\n")
    context = None
    try:
        while True:
            prompt = input("You: ")
            if prompt.lower() == "quit":
                break
            if prompt.lower() == "reset":
                context = None
                print("Context cleared.")
                continue

            full_prompt = prompt if context is None else context + " " + prompt
            input_ids = tokenizer.encode(full_prompt)
            idx = torch.tensor([input_ids], dtype=torch.long).to(device)
            output = model.generate(idx, max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.9)
            generated = output[0].tolist()
            new_tokens = generated[len(input_ids):]
            response = tokenizer.decode(new_tokens)
            print(f"AI: {response}")
            context = full_prompt + " " + response
            if len(tokenizer.encode(context)) > 500:
                context = None
    except KeyboardInterrupt:
        print("\nGoodbye!")
