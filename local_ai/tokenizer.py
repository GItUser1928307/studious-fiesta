"""Tokenizers for text encoding/decoding."""

import json
import re
from collections import Counter

import torch


# ============================================================
# Base tokenizer
# ============================================================

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

    def encode_batch(
        self,
        texts: list[str],
        max_len: int = None,
    ) -> list[list[int]]:
        results = []

        for text in texts:
            ids = self.encode(text)

            if max_len is not None:
                ids = ids[:max_len]

            results.append(ids)

        return results

    def collate(
        self,
        batch: list[list[int]],
        max_len: int,
    ) -> dict:
        padded = []

        for ids in batch:
            ids = ids[:max_len]

            if len(ids) < max_len:
                ids = ids + [
                    self.pad_token_id
                ] * (max_len - len(ids))

            padded.append(ids)

        x = torch.tensor(
            [ids[:-1] for ids in padded],
            dtype=torch.long,
        )

        y = torch.tensor(
            [ids[1:] for ids in padded],
            dtype=torch.long,
        )

        return {
            "input_ids": x,
            "labels": y,
        }

    def info(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id,
        }


# ============================================================
# Word tokenizer
# ============================================================

class WordTokenizer(BaseTokenizer):
    """
    Word-level tokenizer.

    The tokenizer is intentionally case-sensitive for normal words,
    while contractions are normalized into smaller tokens.
    """

    SPECIAL_TOKENS = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<q>",
        "<a>",
    ]

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
        self.word_to_id = {
            str(token): int(token_id)
            for token, token_id in word_to_id.items()
        }

        self.id_to_word = {
            token_id: token
            for token, token_id in self.word_to_id.items()
        }

        self._validate_special_tokens()

        self.vocab_size = len(self.word_to_id)

        self.bos_token_id = self.word_to_id["<bos>"]
        self.eos_token_id = self.word_to_id["<eos>"]
        self.pad_token_id = self.word_to_id["<pad>"]
        self.unk_token_id = self.word_to_id["<unk>"]
        self.q_token_id = self.word_to_id["<q>"]
        self.a_token_id = self.word_to_id["<a>"]

    def _validate_special_tokens(self):
        missing = [
            token
            for token in self.SPECIAL_TOKENS
            if token not in self.word_to_id
        ]

        if missing:
            raise ValueError(
                "Tokenizer vocabulary is missing required "
                f"special tokens: {missing}"
            )

    @classmethod
    def build(
        cls,
        data_file: str,
        min_count: int = 1,
    ) -> "WordTokenizer":

        counter = Counter()

        with open(
            data_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                words = cls.tokenize(line)
                counter.update(words)

        filtered = {
            word
            for word, count in counter.items()
            if count >= min_count
            and word not in cls.SPECIAL_TOKENS
        }

        word_to_id = {}

        # Special tokens always occupy the first IDs.
        for i, token in enumerate(cls.SPECIAL_TOKENS):
            word_to_id[token] = i

        # Deterministic vocabulary ordering.
        for i, word in enumerate(sorted(filtered)):
            word_to_id[word] = (
                i + len(cls.SPECIAL_TOKENS)
            )

        return cls(word_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Tokenize text into words/punctuation.

        <q> and <a> are preserved as special tokens.
        Normal text remains case-sensitive.
        """

        tokens = []

        for word in text.split():

            if word in ("<q>", "<a>"):
                tokens.append(word)
                continue

            lower = word.lower()

            if lower in WordTokenizer.CONTRACTIONS:
                parts = WordTokenizer.CONTRACTIONS[
                    lower
                ].split()

                tokens.extend(parts)
                continue

            # Keep words together while separating punctuation.
            parts = re.findall(
                r"\w+[^\w\s]*|\S",
                word,
            )

            tokens.extend(parts)

        return tokens

    def encode(self, text: str) -> list[int]:
        ids = [
            self.bos_token_id
        ]

        for token in self.tokenize(text):
            ids.append(
                self.word_to_id.get(
                    token,
                    self.unk_token_id,
                )
            )

        ids.append(
            self.eos_token_id
        )

        return ids

    def decode(self, ids: list[int]) -> str:
        words = []

        ignored = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.q_token_id,
            self.a_token_id,
        }

        for token_id in ids:

            if token_id in ignored:
                continue

            token = self.id_to_word.get(
                int(token_id)
            )

            if token is not None:
                words.append(token)

        return " ".join(words)

    def vocab_info(self) -> str:
        lines = [
            f"Vocab size: {self.vocab_size}",
            f"  Special: {self.SPECIAL_TOKENS}",
            (
                "  Words: "
                f"{self.vocab_size - len(self.SPECIAL_TOKENS)}"
            ),
        ]

        return "\n".join(lines)

    def save(self, path: str):
        data = {
            "type": "word",
            "vocab": self.word_to_id,
        }

        _save_json(path, data)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        data = _load_json(path)

        # Support both:
        # 1. New format: {"type": "word", "vocab": {...}}
        # 2. Legacy format: {"<pad>": 0, ...}
        if "vocab" in data:
            data = data["vocab"]

        return cls(data)


# ============================================================
# Character tokenizer
# ============================================================

class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer."""

    SPECIAL_TOKENS = [
        "<pad>",
        "<bos>",
        "<eos>",
    ]

    def __init__(self):
        chars = (
            "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
            "abcdefghijklmnopqrstuvwxyz{|}~"
        )

        self.special_tokens = {
            token: i
            for i, token in enumerate(
                self.SPECIAL_TOKENS
            )
        }

        self.char_to_id = {
            char: i + len(self.SPECIAL_TOKENS)
            for i, char in enumerate(chars)
        }

        self.id_to_char = {
            token_id: char
            for char, token_id in self.char_to_id.items()
        }

        self.id_to_token = {
            token_id: token
            for token, token_id in self.special_tokens.items()
        }

        self.vocab_size = (
            len(self.special_tokens)
            + len(chars)
        )

        self.bos_token_id = (
            self.special_tokens["<bos>"]
        )

        self.eos_token_id = (
            self.special_tokens["<eos>"]
        )

        self.pad_token_id = (
            self.special_tokens["<pad>"]
        )

    def encode(self, text: str) -> list[int]:
        ids = [
            self.bos_token_id
        ]

        for char in text:
            ids.append(
                self.char_to_id.get(
                    char,
                    self.pad_token_id,
                )
            )

        ids.append(
            self.eos_token_id
        )

        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []

        for token_id in ids:
            char = self.id_to_char.get(
                int(token_id)
            )

            if char is not None:
                chars.append(char)

        return "".join(chars)

    def vocab_info(self) -> str:
        lines = [
            f"Vocab size: {self.vocab_size}",
            (
                f"  Special: "
                f"{list(self.special_tokens.keys())}"
            ),
            f"  Chars: {len(self.char_to_id)}",
        ]

        return "\n".join(lines)


# ============================================================
# GPT-2 tokenizer
# ============================================================

class GPT2Tokenizer(BaseTokenizer):
    """BPE tokenizer using tiktoken GPT-2 encoding."""

    def __init__(self):
        import tiktoken

        self.enc = tiktoken.get_encoding(
            "gpt2"
        )

        self.vocab_size = self.enc.n_vocab

        # GPT-2 has an end-of-text token that can safely
        # serve as BOS/EOS/PAD for this simple interface.
        self.bos_token_id = self.enc.eot_token
        self.eos_token_id = self.enc.eot_token
        self.pad_token_id = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def vocab_info(self) -> str:
        return (
            f"Vocab size: {self.vocab_size}\n"
            "  Type: GPT-2 BPE"
        )


# ============================================================
# Whitespace tokenizer
# ============================================================

class WhitespaceTokenizer(BaseTokenizer):
    """Simple whitespace tokenizer."""

    SPECIAL_TOKENS = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<q>",
        "<a>",
    ]

    def __init__(self, word_to_id: dict):
        self.word_to_id = {
            str(token): int(token_id)
            for token, token_id in word_to_id.items()
        }

        self.id_to_word = {
            token_id: token
            for token, token_id in self.word_to_id.items()
        }

        self._validate_special_tokens()

        self.vocab_size = len(
            self.word_to_id
        )

        self.bos_token_id = (
            self.word_to_id["<bos>"]
        )

        self.eos_token_id = (
            self.word_to_id["<eos>"]
        )

        self.pad_token_id = (
            self.word_to_id["<pad>"]
        )

        self.unk_token_id = (
            self.word_to_id["<unk>"]
        )

        self.q_token_id = (
            self.word_to_id["<q>"]
        )

        self.a_token_id = (
            self.word_to_id["<a>"]
        )

    def _validate_special_tokens(self):
        missing = [
            token
            for token in self.SPECIAL_TOKENS
            if token not in self.word_to_id
        ]

        if missing:
            raise ValueError(
                "Tokenizer vocabulary is missing required "
                f"special tokens: {missing}"
            )

    @classmethod
    def build(
        cls,
        data_file: str,
        min_count: int = 1,
    ) -> "WhitespaceTokenizer":

        counter = Counter()

        with open(
            data_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                counter.update(
                    cls.tokenize(line)
                )

        filtered = {
            word
            for word, count in counter.items()
            if count >= min_count
            and word not in cls.SPECIAL_TOKENS
        }

        word_to_id = {}

        for i, token in enumerate(
            cls.SPECIAL_TOKENS
        ):
            word_to_id[token] = i

        for i, word in enumerate(
            sorted(filtered)
        ):
            word_to_id[word] = (
                i + len(cls.SPECIAL_TOKENS)
            )

        return cls(word_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.split()

    def encode(self, text: str) -> list[int]:
        ids = [
            self.bos_token_id
        ]

        for token in self.tokenize(text):
            ids.append(
                self.word_to_id.get(
                    token,
                    self.unk_token_id,
                )
            )

        ids.append(
            self.eos_token_id
        )

        return ids

    def decode(self, ids: list[int]) -> str:
        words = []

        ignored = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.q_token_id,
            self.a_token_id,
        }

        for token_id in ids:

            if token_id in ignored:
                continue

            token = self.id_to_word.get(
                int(token_id)
            )

            if token is not None:
                words.append(token)

        return " ".join(words)

    def vocab_info(self) -> str:
        lines = [
            f"Vocab size: {self.vocab_size}",
            f"  Special: {self.SPECIAL_TOKENS}",
            (
                "  Words: "
                f"{self.vocab_size - len(self.SPECIAL_TOKENS)}"
            ),
        ]

        return "\n".join(lines)

    def save(self, path: str):
        data = {
            "type": "whitespace",
            "vocab": self.word_to_id,
        }

        _save_json(path, data)

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "WhitespaceTokenizer":

        data = _load_json(path)

        if "vocab" in data:
            data = data["vocab"]

        return cls(data)


# ============================================================
# Registry
# ============================================================

TOKENIZER_REGISTRY = {
    "word": WordTokenizer,
    "whitespace": WhitespaceTokenizer,
    "char": CharTokenizer,
    "gpt2": GPT2Tokenizer,
}


# ============================================================
# Tokenizer factory
# ============================================================

def get_tokenizer(
    name: str,
    **kwargs,
):
    """
    Create a tokenizer.

    Word/whitespace tokenizers require a data file when
    building their vocabulary.
    """

    name = str(name).lower().strip()

    if name not in TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer: {name}. "
            f"Available: {list(TOKENIZER_REGISTRY.keys())}"
        )

    cls = TOKENIZER_REGISTRY[name]

    if name in (
        "word",
        "whitespace",
    ):
        data_file = kwargs.get(
            "data_file"
        )

        if not data_file:
            raise ValueError(
                f"The '{name}' tokenizer requires "
                "data_file=..."
            )

        return cls.build(
            data_file
        )

    return cls()


def list_tokenizers():
    return list(
        TOKENIZER_REGISTRY.keys()
    )


# ============================================================
# JSON helpers
# ============================================================

def _save_json(
    path: str,
    data: dict,
):
    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def _load_json(path: str) -> dict:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid tokenizer file: {path}"
        )

    return data


# ============================================================
# Save tokenizer
# ============================================================

def save_tokenizer(
    tokenizer,
    path: str,
):
    """
    Save a tokenizer in a portable JSON format.

    Word and whitespace tokenizers store their complete
    vocabulary, so loading does not require rebuilding the
    vocabulary from the training dataset.
    """

    type_name = None

    for name, cls in TOKENIZER_REGISTRY.items():
        if isinstance(
            tokenizer,
            cls,
        ):
            type_name = name
            break

    if type_name is None:
        raise TypeError(
            "Unsupported tokenizer type: "
            f"{type(tokenizer).__name__}"
        )

    if type_name in (
        "word",
        "whitespace",
    ):
        data = {
            "type": type_name,
            "vocab": tokenizer.word_to_id,
        }

    elif type_name == "char":
        data = {
            "type": "char",
        }

    elif type_name == "gpt2":
        data = {
            "type": "gpt2",
        }

    else:
        raise ValueError(
            f"Unsupported tokenizer type: {type_name}"
        )

    _save_json(
        path,
        data,
    )


# ============================================================
# Load tokenizer
# ============================================================

def load_tokenizer(
    path: str,
    data_file: str = None,
):
    """
    Load a tokenizer from a saved JSON file.

    For word/whitespace tokenizers, the saved vocabulary is
    always preferred over rebuilding from the dataset.

    The data_file argument is retained for compatibility with
    older code and is only used when a tokenizer file does not
    contain a saved vocabulary.
    """

    data = _load_json(path)

    type_name = str(
        data.get(
            "type",
            "word",
        )
    ).lower()

    if type_name not in TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer type '{type_name}' "
            f"in {path}. "
            f"Available: "
            f"{list(TOKENIZER_REGISTRY.keys())}"
        )

    cls = TOKENIZER_REGISTRY[
        type_name
    ]

    # Word/whitespace tokenizer with saved vocabulary.
    if type_name in (
        "word",
        "whitespace",
    ):

        vocab = data.get(
            "vocab"
        )

        if vocab is not None:
            return cls(vocab)

        # Legacy tokenizer files may contain the vocabulary
        # directly at the root.
        if all(
            isinstance(key, str)
            for key in data.keys()
        ):
            special_keys = set(
                cls.SPECIAL_TOKENS
            )

            if special_keys.issubset(
                data.keys()
            ):
                return cls(data)

        # Last-resort compatibility path.
        if data_file:
            return cls.build(
                data_file
            )

        raise ValueError(
            f"Tokenizer file '{path}' does not contain "
            "a saved vocabulary and no data_file was supplied."
        )

    # Character tokenizer has a deterministic vocabulary.
    if type_name == "char":
        return cls()

    # GPT-2 tokenizer has a deterministic external vocabulary.
    if type_name == "gpt2":
        return cls()

    raise ValueError(
        f"Unable to load tokenizer type: {type_name}"
    )
