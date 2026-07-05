import os
import re
import torch
from model import create_model
from tokenizer import load_tokenizer, get_tokenizer, WordTokenizer
from config import ModelConfig


def load_model_and_tokenizer(model_path: str, device: str = "cpu"):
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

    tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer_name = checkpoint.get("tokenizer", "word")
        data_file = os.path.join(os.path.dirname(os.path.dirname(model_path)), "quick_train_data.txt")
        tokenizer = get_tokenizer(tokenizer_name, data_file=data_file)
    return model, tokenizer


def clean_response(text: str) -> str:
    stop_pattern = re.compile(r"\b(what is|what are|what was|what were|how do|how does|how did|how is|how are|why do|why does|why is|why are|can you|do you|are you|tell me|what do|what can|what makes|what is a|what is an|what is the)\b", re.IGNORECASE)
    match = stop_pattern.search(text)
    if match:
        text = text[:match.start()]
    junk = ["<bos>", "<eos>", "<pad>", "<unk>", "<q>", "<a>"]
    for j in junk:
        text = text.replace(j, " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text = text.rstrip(" ,;:-")
    return text


def generate_text(model, tokenizer, prompt: str, max_new: int = 100, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95, device: str = "cpu"):
    q_tokens = [t for t in tokenizer.tokenize(prompt) if t not in ("<q>", "<a>")]
    input_ids = [tokenizer.bos_token_id, tokenizer.q_token_id]
    input_ids += [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in q_tokens]
    input_ids.append(tokenizer.a_token_id)

    idx = torch.tensor([input_ids], dtype=torch.long).to(device)
    output = model.generate(idx, max_new, temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=tokenizer.eos_token_id)
    generated = output[0].tolist()
    new_tokens = generated[len(input_ids):]
    raw = tokenizer.decode(new_tokens)
    return clean_response(raw)


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
            q_tokens = [t for t in tokenizer.tokenize(full_prompt) if t not in ("<q>", "<a>")]
            input_ids = [tokenizer.bos_token_id, tokenizer.q_token_id]
            input_ids += [tokenizer.word_to_id.get(t, tokenizer.unk_token_id) for t in q_tokens]
            input_ids.append(tokenizer.a_token_id)

            idx = torch.tensor([input_ids], dtype=torch.long).to(device)
            output = model.generate(idx, max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
            generated = output[0].tolist()
            new_tokens = generated[len(input_ids):]
            response = clean_response(tokenizer.decode(new_tokens))
            print(f"AI: {response}")
            context = prompt + " " + response
            if len(tokenizer.encode(context)) > 500:
                context = None
    except KeyboardInterrupt:
        print("\nGoodbye!")
