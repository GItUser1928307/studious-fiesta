import tiktoken
import torch


class GPT2Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.bos_token_id = self.enc.eot_token
        self.eos_token_id = self.enc.eot_token
        self.pad_token_id = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def encode_batch(self, texts: list[str], max_len: int = None) -> torch.Tensor:
        encoded = []
        for text in texts:
            ids = self.encode(text)
            if max_len:
                ids = ids[:max_len - 1]
            encoded.append(ids)
        return encoded

    def collate(self, batch: list[list[int]], max_len: int) -> dict:
        input_ids = []
        for ids in batch:
            ids = ids[:max_len]
            input_ids.append(ids + [self.pad_token_id] * (max_len - len(ids)))
        x = torch.tensor([ids[:-1] for ids in input_ids], dtype=torch.long)
        y = torch.tensor([ids[1:] for ids in input_ids], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class CharTokenizer:
    def __init__(self):
        self.chars = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        self.vocab_size = len(self.chars) + 3
        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}
        self.stoi["<bos>"] = 0
        self.stoi["<eos>"] = 1
        self.stoi["<pad>"] = 2
        self.itos = {v: k for k, v in self.stoi.items()}
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                ids.append(self.stoi.get("<pad>"))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i >= 3:
                chars.append(self.itos.get(i, ""))
        return "".join(chars)

    def collate(self, batch: list[list[int]], max_len: int) -> dict:
        input_ids = []
        for ids in batch:
            ids = ids[:max_len]
            input_ids.append(ids + [self.pad_token_id] * (max_len - len(ids)))
        x = torch.tensor([ids[:-1] for ids in input_ids], dtype=torch.long)
        y = torch.tensor([ids[1:] for ids in input_ids], dtype=torch.long)
        return {"input_ids": x, "labels": y}
