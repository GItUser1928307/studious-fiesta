"""Tokenizers for text encoding/decoding."""
import torch


class BaseTokenizer:
    """Shared interface for all tokenizers."""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def encode_batch(self, texts: list[str], max_len: int = None) -> list[list[int]]:
        results = []
        for text in texts:
            ids = self.encode(text)
            if max_len:
                ids = ids[:max_len]
            results.append(ids)
        return results

    def collate(self, batch: list[list[int]], max_len: int) -> dict:
        padded = []
        for ids in batch:
            ids = ids[:max_len]
            padded.append(ids + [self.pad_token_id] * (max_len - len(ids)))
        x = torch.tensor([ids[:-1] for ids in padded], dtype=torch.long)
        y = torch.tensor([ids[1:] for ids in padded], dtype=torch.long)
        return {"input_ids": x, "labels": y}

    def info(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id,
        }


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer. Maps each character to a unique ID."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

    def __init__(self):
        chars = (
            "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
            "abcdefghijklmnopqrstuvwxyz{|}~"
        )
        self.special_tokens = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.char_to_id = {ch: i + len(self.SPECIAL_TOKENS) for i, ch in enumerate(chars)}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        self.vocab_size = len(self.special_tokens) + len(chars)
        self.bos_token_id = self.special_tokens["<bos>"]
        self.eos_token_id = self.special_tokens["<eos>"]
        self.pad_token_id = self.special_tokens["<pad>"]

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for ch in text:
            ids.append(self.char_to_id.get(ch, self.pad_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i in self.id_to_char:
                chars.append(self.id_to_char[i])
        return "".join(chars)

    def vocab_info(self) -> str:
        lines = [f"Vocab size: {self.vocab_size}"]
        lines.append(f"  Special: {list(self.special_tokens.keys())}")
        lines.append(f"  Chars: {len(self.char_to_id)}")
        return "\n".join(lines)


class GPT2Tokenizer(BaseTokenizer):
    """BPE tokenizer using tiktoken (GPT-2 encoding)."""

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.bos_token_id = self.enc.eot_token
        self.eos_token_id = self.enc.eot_token
        self.pad_token_id = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)
