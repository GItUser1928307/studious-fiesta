"""Tokenizers for text encoding/decoding."""
import torch
import re
from collections import Counter


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


class WordTokenizer(BaseTokenizer):
    """Word-level tokenizer. Splits on whitespace, builds vocab from data."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<q>", "<a>"]

    CONTRACTIONS = {
        "don't": "do n't",
        "doesn't": "does n't",
        "didn't": "did n't",
        "isn't": "is n't",
        "wasn't": "was n't",
        "weren't": "were n't",
        "won't": "will n't",
        "wouldn't": "would n't",
        "can't": "can n't",
        "couldn't": "could n't",
        "shouldn't": "should n't",
        "haven't": "have n't",
        "hasn't": "has n't",
        "hadn't": "had n't",
        "aren't": "are n't",
        "i'm": "i 'm",
        "you're": "you 're",
        "he's": "he 's",
        "she's": "she 's",
        "it's": "it 's",
        "we're": "we 're",
        "they're": "they 're",
        "i've": "i 've",
        "you've": "you 've",
        "we've": "we 've",
        "they've": "they 've",
        "i'll": "i 'll",
        "you'll": "you 'll",
        "he'll": "he 'll",
        "she'll": "she 'll",
        "we'll": "we 'll",
        "they'll": "they 'll",
        "i'd": "i 'd",
        "you'd": "you 'd",
        "he'd": "he 'd",
        "she'd": "she 'd",
        "we'd": "we 'd",
        "they'd": "they 'd",
        "that's": "that 's",
        "who's": "who 's",
        "what's": "what 's",
        "where's": "where 's",
        "when's": "when 's",
        "how's": "how 's",
        "there's": "there 's",
        "here's": "here 's",
        "let's": "let 's",
    }

    def __init__(self, word_to_id: dict):
        self.word_to_id = word_to_id
        self.id_to_word = {v: k for k, v in word_to_id.items()}
        self.vocab_size = len(word_to_id)
        self.bos_token_id = self.word_to_id["<bos>"]
        self.eos_token_id = self.word_to_id["<eos>"]
        self.pad_token_id = self.word_to_id["<pad>"]
        self.unk_token_id = self.word_to_id["<unk>"]
        self.q_token_id = self.word_to_id["<q>"]
        self.a_token_id = self.word_to_id["<a>"]

    @classmethod
    def build(cls, data_file: str, min_count: int = 1) -> "WordTokenizer":
        counter = Counter()
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = cls.tokenize(line)
                counter.update(words)

        filtered = {w for w, c in counter.items() if c >= min_count and w not in cls.SPECIAL_TOKENS}

        word_to_id = {}
        for i, tok in enumerate(cls.SPECIAL_TOKENS):
            word_to_id[tok] = i
        for i, word in enumerate(sorted(filtered)):
            word_to_id[word] = i + len(cls.SPECIAL_TOKENS)

        return cls(word_to_id)

    @staticmethod
    def tokenize(text: str) -> list:
        # Case-sensitive: "Hello" and "hello" are different tokens.
        tokens = []
        for word in text.split():
            if word in ("<q>", "<a>"):
                tokens.append(word)
                continue
            lower = word.lower()
            if lower in WordTokenizer.CONTRACTIONS:
                parts = WordTokenizer.CONTRACTIONS[lower].split()
                tokens.extend(parts)
            else:
                parts = re.findall(r"\w+[^\w\s]*|\S", word)
                tokens.extend(parts)
        return tokens

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for token in self.tokenize(text):
            ids.append(self.word_to_id.get(token, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        words = []
        for i in ids:
            if i in self.id_to_word and i not in (
                self.bos_token_id, self.eos_token_id, self.pad_token_id,
                self.q_token_id, self.a_token_id
            ):
                words.append(self.id_to_word[i])
        return " ".join(words)

    def vocab_info(self) -> str:
        lines = [f"Vocab size: {self.vocab_size}"]
        lines.append(f"  Special: {self.SPECIAL_TOKENS}")
        lines.append(f"  Words: {self.vocab_size - len(self.SPECIAL_TOKENS)}")
        return "\n".join(lines)

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        import json
        with open(path, encoding="utf-8") as f:
            word_to_id = json.load(f)
        return cls(word_to_id)


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


class WhitespaceTokenizer(BaseTokenizer):
    """Simple whitespace tokenizer. Fast, no punctuation splitting."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<q>", "<a>"]

    def __init__(self, word_to_id: dict):
        self.word_to_id = word_to_id
        self.id_to_word = {v: k for k, v in word_to_id.items()}
        self.vocab_size = len(word_to_id)
        self.bos_token_id = self.word_to_id["<bos>"]
        self.eos_token_id = self.word_to_id["<eos>"]
        self.pad_token_id = self.word_to_id["<pad>"]
        self.unk_token_id = self.word_to_id["<unk>"]
        self.q_token_id = self.word_to_id["<q>"]
        self.a_token_id = self.word_to_id["<a>"]

    @classmethod
    def build(cls, data_file: str, min_count: int = 1) -> "WhitespaceTokenizer":
        counter = Counter()
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = cls.tokenize(line)
                counter.update(words)
        filtered = {w for w, c in counter.items() if c >= min_count and w not in cls.SPECIAL_TOKENS}
        word_to_id = {}
        for i, tok in enumerate(cls.SPECIAL_TOKENS):
            word_to_id[tok] = i
        for i, word in enumerate(sorted(filtered)):
            word_to_id[word] = i + len(cls.SPECIAL_TOKENS)
        return cls(word_to_id)

    @staticmethod
    def tokenize(text: str) -> list:
        return text.split()

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for token in self.tokenize(text):
            ids.append(self.word_to_id.get(token, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        words = []
        for i in ids:
            if i in self.id_to_word and i not in (
                self.bos_token_id, self.eos_token_id, self.pad_token_id,
                self.q_token_id, self.a_token_id
            ):
                words.append(self.id_to_word[i])
        return " ".join(words)

    def vocab_info(self) -> str:
        lines = [f"Vocab size: {self.vocab_size}"]
        lines.append(f"  Special: {self.SPECIAL_TOKENS}")
        lines.append(f"  Words: {self.vocab_size - len(self.SPECIAL_TOKENS)}")
        return "\n".join(lines)

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WhitespaceTokenizer":
        import json
        with open(path, encoding="utf-8") as f:
            word_to_id = json.load(f)
        return cls(word_to_id)


TOKENIZER_REGISTRY = {
    "word": WordTokenizer,
    "whitespace": WhitespaceTokenizer,
    "char": CharTokenizer,
    "gpt2": GPT2Tokenizer,
}


def get_tokenizer(name: str, **kwargs):
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer: {name}. Available: {list(TOKENIZER_REGISTRY.keys())}")
    cls = TOKENIZER_REGISTRY[name]
    if name in ("word", "whitespace"):
        return cls.build(kwargs["data_file"])
    return cls()


def list_tokenizers():
    return list(TOKENIZER_REGISTRY.keys())


def save_tokenizer(tokenizer, path: str):
    import json
    type_name = None
    for name, cls in TOKENIZER_REGISTRY.items():
        if isinstance(tokenizer, cls):
            type_name = name
            break
    if type_name in ("word", "whitespace"):
        data = {"type": type_name, "vocab": tokenizer.word_to_id}
    else:
        data = {"type": type_name or "char"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_tokenizer(path: str, data_file: str = None):
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    type_name = data.get("type", "word")
    if type_name in ("word", "whitespace") and "vocab" in data:
        return TOKENIZER_REGISTRY[type_name](data["vocab"])
    return get_tokenizer(type_name, data_file=data_file)