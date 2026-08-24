import os
import re
import torch

from model import create_model
from tokenizer import load_tokenizer, get_tokenizer
from config import ModelConfig


# ============================================================
# Model + tokenizer loading
# ============================================================

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cpu",
):
    from config import SMALL_CONFIG

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    config_data = checkpoint.get("config")

    if config_data is None:
        config = SMALL_CONFIG

    elif isinstance(config_data, dict):
        config = ModelConfig(**config_data)

    else:
        config = config_data

    model = create_model(config)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Load the tokenizer saved alongside the checkpoint.
    # --------------------------------------------------------

    tokenizer_path = os.path.join(
        os.path.dirname(model_path),
        "tokenizer.json",
    )

    if os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(
            tokenizer_path
        )

    else:
        tokenizer_name = checkpoint.get(
            "tokenizer",
            "word",
        )

        data_file = os.path.join(
            os.path.dirname(
                os.path.dirname(model_path)
            ),
            "quick_train_data.txt",
        )

        tokenizer = get_tokenizer(
            tokenizer_name,
            data_file=data_file,
        )

    return model, tokenizer


# ============================================================
# Response cleanup
# ============================================================

def clean_response(text: str) -> str:
    """
    Clean generated text without aggressively destroying
    normal answers.

    Removes special training tokens and obvious accidental
    question/prompt continuation.
    """

    stop_pattern = re.compile(
        r"\b("
        r"what is|what are|what was|what were|"
        r"how do|how does|how did|how is|how are|"
        r"why do|why does|why is|why are|"
        r"can you|do you|are you|tell me|"
        r"what do|what can|what makes|"
        r"what is a|what is an|what is the"
        r")\b",
        re.IGNORECASE,
    )

    match = stop_pattern.search(text)

    if match:
        text = text[:match.start()]

    junk_tokens = [
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<q>",
        "<a>",
    ]

    for token in junk_tokens:
        text = text.replace(
            token,
            " ",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if text and text[-1] not in ".!?":
        text = text.rstrip(
            " ,;:-"
        )

    return text


# ============================================================
# Prompt encoding
# ============================================================

def build_input_ids(
    tokenizer,
    prompt: str,
):
    """
    Convert a normal user prompt into the same
    <bos> <q> question <a> format used during training.
    """

    tokens = [
        token
        for token in tokenizer.tokenize(prompt)
        if token not in ("<q>", "<a>")
    ]

    input_ids = [
        tokenizer.bos_token_id,
        tokenizer.q_token_id,
    ]

    input_ids.extend(
        tokenizer.word_to_id.get(
            token,
            tokenizer.unk_token_id,
        )
        for token in tokens
    )

    input_ids.append(
        tokenizer.a_token_id
    )

    return input_ids


# ============================================================
# Text generation
# ============================================================

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    device: str = "cpu",
):
    """
    Generate one response from the model.
    """

    input_ids = build_input_ids(
        tokenizer,
        prompt,
    )

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )

    generated = output[0].tolist()

    new_tokens = generated[
        len(input_ids):
    ]

    raw_response = tokenizer.decode(
        new_tokens
    )

    return clean_response(
        raw_response
    )


# ============================================================
# Interactive generation
# ============================================================

def interactive(
    model,
    tokenizer,
    device: str,
):
    print("\n=== Interactive Generation ===")
    print("Type 'quit' to exit.")
    print("Type 'reset' to clear context.")
    print("Press Ctrl+C to stop generation.\n")

    context = None

    try:
        while True:
            prompt = input("You: ").strip()

            if not prompt:
                continue

            if prompt.lower() == "quit":
                break

            if prompt.lower() == "reset":
                context = None
                print("Context cleared.")
                continue

            # ------------------------------------------------
            # Build conversational context.
            # ------------------------------------------------

            if context is None:
                full_prompt = prompt
            else:
                full_prompt = (
                    context
                    + " "
                    + prompt
                )

            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                max_new=80,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.05,
                device=device,
            )

            if not response:
                response = "I don't know."

            print(
                f"AI: {response}"
            )

            # ------------------------------------------------
            # Preserve conversational context.
            # ------------------------------------------------

            context = (
                full_prompt
                + " "
                + response
            )

            # Prevent context from growing indefinitely.
            try:
                context_tokens = tokenizer.encode(
                    context
                )

                if len(context_tokens) > 500:
                    context = None

            except Exception:
                # If the tokenizer does not support the
                # expected encode() behavior, safely reset.
                context = None

    except KeyboardInterrupt:
        print("\nGoodbye!")


# ============================================================
# Optional command-line entry point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate text with the trained model."
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to generate from.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu, cuda, or xla.",
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=100,
        help="Maximum number of new tokens.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus/top-p sampling.",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Penalty applied to previously generated tokens.",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    parent_dir = os.path.dirname(
        script_dir
    )

    default_model = os.path.join(
        parent_dir,
        "quick_ckpt",
        "best.pt",
    )

    model_path = (
        args.model
        if args.model
        else default_model
    )

    if args.device:
        device = args.device

    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if not os.path.exists(model_path):
        print(
            f"ERROR: Model checkpoint not found:\n"
            f"{model_path}"
        )
        return

    print(
        f"Loading model: {model_path}"
    )

    print(
        f"Device: {device}"
    )

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        device,
    )

    print(
        f"Parameters: "
        f"{model.count_params():,}"
    )

    print(
        f"Vocabulary: "
        f"{tokenizer.vocab_size}"
    )

    if args.prompt is not None:
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )

        print(
            f"\nAI: {response}"
        )

    else:
        interactive(
            model,
            tokenizer,
            device,
        )


if __name__ == "__main__":
    main()
