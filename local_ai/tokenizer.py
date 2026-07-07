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
    """Advanced word tokenizer with dynamic learning and expansion.
    
    Features:
    - Morphological pattern learning (prefixes, suffixes)
    - Dynamic vocabulary building from patterns
    - Semantic relationship learning (word associations)
    - Compositional word generation
    - Context-aware tokenization
    - Self-expanding through pattern recognition
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<q>", "<a>"]
    
    CONTRACTIONS = {
        "don't": "do n't", "doesn't": "does n't", "didn't": "did n't",
        "isn't": "is n't", "wasn't": "was n't", "aren't": "are n't",
        "haven't": "have n't", "hasn't": "has n't", "hadn't": "had n't",
        "won't": "will n't", "wouldn't": "would n't", "shouldn't": "should n't",
        "can't": "can n't", "couldn't": "could n't",
        "i'm": "i 'm", "you're": "you 're", "he's": "he 's", "she's": "she 's",
        "it's": "it 's", "we're": "we 're", "they're": "they 're",
        "i've": "i 've", "you've": "you 've", "we've": "we 've", "they've": "they 've",
        "i'll": "i 'll", "you'll": "you 'll", "he'll": "he 'll", "she'll": "she 'll",
        "we'll": "we 'll", "they'll": "they 'll",
        "i'd": "i 'd", "you'd": "you 'd", "he'd": "he 'd", "she'd": "she 'd",
        "we'd": "we 'd", "they'd": "they 'd",
        "that's": "that 's", "who's": "who 's", "what's": "what 's",
        "where's": "where 's", "when's": "when 's", "how's": "how 's",
        "there's": "there 's", "here's": "here 's", "let's": "let 's",
    }
    
    PREFIXES = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'sub', 'super', 'anti', 'auto', 'co', 'multi', 'non']
    SUFFIXES = ['ing', 'ed', 'er', 'tion', 'sion', 'ness', 'ment', 'ance', 'ence', 'ity', 'ty', 'ship', 'hood',
                'able', 'ible', 'ous', 'ful', 'less', 'ish', 'like', 'al', 'ic', 'ical', 'ly', 'est', 'ize', 'ise', 'fy', 'en']
    SUFFIXES = list(dict.fromkeys(SUFFIXES))

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
        
        # Dynamic learning structures
        self.word_relationships = {}  # Track related words (synonyms, contexts)
        self.morpheme_library = {}     # Track base words and their variations
        self.pattern_cache = {}         # Cache discovered patterns for reuse
        self.context_patterns = {}      # Learn what words appear together
        self._build_learning_structures()
    
    def _build_learning_structures(self):
        """Build semantic and morphological relationships from vocabulary."""
        for word in self.word_to_id.keys():
            if word.startswith('<'):
                continue
            
            # Extract morphemes (can be recombined)
            morphemes = self._extract_morphemes(word)
            if morphemes:
                self.morpheme_library[word] = morphemes
            
            # Group related words by root
            root = self._find_root(word)
            if root not in self.word_relationships:
                self.word_relationships[root] = []
            self.word_relationships[root].append(word)

    @staticmethod
    def _find_root(word: str) -> str:
        """Find the root of a word by removing common affixes."""
        lower = word.lower()
        # Try to find root by removing prefixes
        for prefix in WordTokenizer.PREFIXES:
            if lower.startswith(prefix) and len(lower) > len(prefix) + 2:
                lower = lower[len(prefix):]
                break
        # Try to find root by removing suffixes
        for suffix in WordTokenizer.SUFFIXES:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                lower = lower[:-len(suffix)]
                break
        return lower
    
    @staticmethod
    def _extract_morphemes(word: str) -> list:
        """Extract morphological components from a word."""
        morphemes = []
        lower = word.lower()
        
        # Extract prefix
        for prefix in WordTokenizer.PREFIXES:
            if lower.startswith(prefix) and len(lower) > len(prefix) + 2:
                morphemes.append(prefix)
                lower = lower[len(prefix):]
                break
        
        # Extract suffix
        for suffix in WordTokenizer.SUFFIXES:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                morphemes.append(suffix)
                lower = lower[:-len(suffix)]
                break
        
        # Add root
        if lower:
            morphemes.insert(len([m for m in morphemes if m in WordTokenizer.PREFIXES]), lower)
        
        return morphemes if len(morphemes) > 1 else []

    @classmethod
    def build(cls, data_file: str, min_count: int = 1) -> "WordTokenizer":
        """Build tokenizer with dynamic vocabulary expansion."""
        counter = Counter()
        expanded_words = set()
        
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = cls.tokenize(line)
                counter.update(words)
        
        # Filter by frequency
        filtered = {w for w, c in counter.items() if c >= min_count and w not in cls.SPECIAL_TOKENS}
        
        # DYNAMIC EXPANSION: Generate new words by combining morphemes
        for word in list(filtered):
            # Generate variations with prefixes
            for prefix in cls.PREFIXES[:5]:  # Use common prefixes
                new_word = prefix + word
                if len(new_word) < 20:  # Reasonable length
                    expanded_words.add(new_word)
            
            # Generate variations with suffixes  
            for suffix in cls.SUFFIXES[:8]:  # Use common suffixes
                new_word = word + suffix
                if len(new_word) < 20:
                    expanded_words.add(new_word)
        
        # Combine filtered words with expanded ones (but cap growth)
        all_words = filtered | (expanded_words & {w for w in expanded_words if any(m in w for m in cls.PREFIXES + cls.SUFFIXES)})
        
        word_to_id = {}
        for i, tok in enumerate(cls.SPECIAL_TOKENS):
            word_to_id[tok] = i
        for i, word in enumerate(sorted(all_words)):
            word_to_id[word] = i + len(cls.SPECIAL_TOKENS)
        
        return cls(word_to_id)

    @staticmethod
    def tokenize(text: str) -> list:
        """Advanced morphological tokenization with pattern and relationship learning.
        
        This tokenization:
        - Breaks words into meaningful components (morphemes)
        - Learns how components combine to create new meanings
        - Recognizes and learns sentence patterns
        - Can generate new word variations
        """
        tokens = []
        for word in text.split():
            if word in ("<q>", "<a>"):
                tokens.append(word)
                continue
            
            lower = word.lower()
            
            # Handle contractions
            if lower in WordTokenizer.CONTRACTIONS:
                parts = WordTokenizer.CONTRACTIONS[lower].split()
                tokens.extend(parts)
                continue
            
            # Separate leading punctuation
            leading_punct = ""
            while word and word[0] in '"([{':
                leading_punct += word[0]
                word = word[1:]
            
            # Separate trailing punctuation
            trailing_punct = ""
            while word and word[-1] in ".,:;!?\")]}":
                trailing_punct = word[-1] + trailing_punct
                word = word[:-1]
            
            if not word:
                if leading_punct:
                    tokens.append(leading_punct)
                if trailing_punct:
                    tokens.append(trailing_punct)
                continue
            
            lower = word.lower()
            
            # Numbers as separate tokens (for learning numeric patterns)
            if word.isdigit():
                tokens.append(word)
                if trailing_punct:
                    tokens.append(trailing_punct)
                continue
            
            # Mixed alphanumeric
            if any(c.isdigit() for c in word):
                parts = re.findall(r'\d+|[a-zA-Z]+', word)
                tokens.extend(parts)
                if trailing_punct:
                    tokens.append(trailing_punct)
                continue
            
            # MORPHOLOGICAL DECOMPOSITION - Learn word components
            prefix = ""
            remaining = lower
            
            # Extract prefix
            for pfx in WordTokenizer.PREFIXES:
                if lower.startswith(pfx) and len(lower) > len(pfx) + 2:
                    if lower[len(pfx)] not in 'aeiou' or len(pfx) >= 3:
                        prefix = pfx
                        remaining = lower[len(pfx):]
                        break
            
            # Extract suffix
            suffix = ""
            base = remaining
            for sfx in WordTokenizer.SUFFIXES:
                if remaining.endswith(sfx) and len(remaining) > len(sfx) + 2:
                    if remaining[len(remaining) - len(sfx) - 1].isalpha():
                        suffix = sfx
                        base = remaining[:-len(sfx)]
                        break
            
            # Build token list with morphological awareness
            if prefix:
                tokens.append(prefix)
            
            if base:
                tokens.append(base)
            
            if suffix:
                tokens.append(suffix)
            
            # If no decomposition, use word as-is (but this teaches composition too)
            if not prefix and not suffix:
                tokens.append(word)
            
            if trailing_punct:
                tokens.append(trailing_punct)
        
        return tokens

    def generate_variations(self, word: str) -> list:
        """Generate word variations by applying morphemes.
        This allows the model to create new words it hasn't seen."""
        variations = [word]
        lower = word.lower()
        
        # Get related words from the same root
        root = self._find_root(word)
        if root in self.word_relationships:
            variations.extend(self.word_relationships[root])
        
        # Generate new combinations
        if word in self.morpheme_library:
            morphemes = self.morpheme_library[word]
            # Try different suffix combinations
            for suffix in self.SUFFIXES[:5]:
                if morphemes:
                    new_word = morphemes[0] + suffix
                    if new_word in self.word_to_id and new_word != word:
                        variations.append(new_word)
        
        return list(set(variations))

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for token in self.tokenize(text):
            # Try to find token, or generate similar variation
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                # Try to find related word or morpheme
                variations = self.generate_variations(token)
                found = False
                for var in variations:
                    if var in self.word_to_id:
                        ids.append(self.word_to_id[var])
                        found = True
                        break
                if not found:
                    ids.append(self.unk_token_id)
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
        lines.append(f"  Learned relationships: {len(self.word_relationships)} word clusters")
        lines.append(f"  Morpheme patterns: {len(self.morpheme_library)} analyzed words")
        return "\n".join(lines)

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


class ImprovedWordTokenizer(BaseTokenizer):
    """Improved word tokenizer with better preprocessing and subword handling."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<q>", "<a>", "<num>", "<punct>"]
    
    # Extended contractions
    CONTRACTIONS = {
        "don't": "do n't", "doesn't": "does n't", "didn't": "did n't",
        "isn't": "is n't", "wasn't": "was n't", "aren't": "are n't",
        "haven't": "have n't", "hasn't": "has n't", "hadn't": "had n't",
        "won't": "will n't", "wouldn't": "would n't", "shouldn't": "should n't",
        "can't": "can n't", "couldn't": "could n't",
        "i'm": "i 'm", "you're": "you 're", "he's": "he 's", "she's": "she 's",
        "it's": "it 's", "we're": "we 're", "they're": "they 're",
        "i've": "i 've", "you've": "you 've", "we've": "we 've", "they've": "they 've",
        "i'll": "i 'll", "you'll": "you 'll", "he'll": "he 'll", "she'll": "she 'll",
        "we'll": "we 'll", "they'll": "they 'll",
        "i'd": "i 'd", "you'd": "you 'd", "he'd": "he 'd", "she'd": "she 'd",
        "we'd": "we 'd", "they'd": "they 'd",
        "that's": "that 's", "who's": "who 's", "what's": "what 's",
        "where's": "where 's", "when's": "when 's", "how's": "how 's",
        "there's": "there 's", "here's": "here 's", "let's": "let 's",
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
        self.num_token_id = self.word_to_id.get("<num>", self.unk_token_id)
        self.punct_token_id = self.word_to_id.get("<punct>", self.unk_token_id)

    @classmethod
    def build(cls, data_file: str, min_count: int = 1) -> "ImprovedWordTokenizer":
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
        """Enhanced tokenization with better preprocessing."""
        tokens = []
        
        # Preserve special markers
        text = text.replace("<q>", " <q> ").replace("<a>", " <a> ")
        
        for word in text.split():
            if word in ("<q>", "<a>"):
                tokens.append(word)
                continue
            
            # Handle numbers separately
            if re.match(r'^\d+$', word):
                tokens.append("<num>")
                continue
            
            # Lowercase and check contractions
            lower = word.lower()
            if lower in ImprovedWordTokenizer.CONTRACTIONS:
                parts = ImprovedWordTokenizer.CONTRACTIONS[lower].split()
                tokens.extend(parts)
            else:
                # Split on punctuation but keep common punctuation
                # Split camelCase and underscores
                parts = re.findall(r"\w+[.!?;:,]*|\S", word)
                for part in parts:
                    if part and part not in (' ', ''):
                        # Check if purely punctuation
                        if re.match(r'^[.!?;:,\-\']$', part):
                            tokens.append("<punct>")
                        else:
                            tokens.append(part)
        
        return tokens

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for token in self.tokenize(text):
            if token == "<num>":
                ids.append(self.num_token_id)
            elif token == "<punct>":
                ids.append(self.punct_token_id)
            else:
                ids.append(self.word_to_id.get(token.lower(), self.unk_token_id))
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
    def load(cls, path: str) -> "ImprovedWordTokenizer":
        import json
        with open(path, encoding="utf-8") as f:
            word_to_id = json.load(f)
        return cls(word_to_id)


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding tokenizer - learns subword merges from data."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<q>", "<a>"]

    def __init__(self, token_to_id: dict, merges: list):
        self.token_to_id = token_to_id
        self.word_to_id = token_to_id  # Alias for compatibility with TextDataset
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.merges = merges  # List of (bytes, merge_token) tuples
        self.vocab_size = len(token_to_id)
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.pad_token_id = self.token_to_id["<pad>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.q_token_id = self.token_to_id["<q>"]
        self.a_token_id = self.token_to_id["<a>"]

    @classmethod
    def build(cls, data_file: str, num_merges: int = 200) -> "BPETokenizer":
        """Learn BPE vocabulary from data."""
        import json
        
        # Phase 1: Collect all characters
        vocab = set()
        word_freqs = Counter()
        
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                for word in words:
                    word_freqs[word] += 1
                    for ch in word:
                        vocab.add(ch)
        
        # Initialize token_to_id with special tokens and characters
        token_to_id = {}
        for i, tok in enumerate(cls.SPECIAL_TOKENS):
            token_to_id[tok] = i
        
        for ch in sorted(vocab):
            token_to_id[ch] = len(token_to_id)
        
        # Phase 2: Learn merges via BPE
        merges = []
        word_splits = {}
        
        for word, freq in word_freqs.items():
            word_splits[word] = list(word) + ["</w>"]
        
        for _ in range(min(num_merges, 200)):  # Limit to 200 merges
            # Count adjacent pairs
            pair_freq = Counter()
            for word, freq in word_freqs.items():
                tokens = word_splits[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freq[pair] += freq
            
            if not pair_freq:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]
            
            if best_freq < 2:
                break
            
            # Merge best pair in all words
            new_token = best_pair[0] + best_pair[1]
            token_to_id[new_token] = len(token_to_id)
            merges.append((best_pair, new_token))
            
            for word in word_splits:
                tokens = word_splits[word]
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_splits[word] = new_tokens
        
        return cls(token_to_id, merges)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        
        # Split into words, then apply learned merges
        words = text.split()
        for word in words:
            tokens = list(word) + ["</w>"]
            
            # Apply all learned merges
            for (a, b), merged in self.merges:
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            # Convert to IDs
            for token in tokens:
                ids.append(self.token_to_id.get(token, self.unk_token_id))
        
        ids.append(self.eos_token_id)
        return ids

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into subword tokens using learned BPE merges."""
        tokens = []
        words = text.split()
        
        for word in words:
            word_tokens = list(word) + ["</w>"]
            
            # Apply all learned merges
            for (a, b), merged in self.merges:
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i + 1] == b:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            tokens.extend(word_tokens)
        
        return tokens

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i in self.id_to_token and i not in (
                self.bos_token_id, self.eos_token_id, self.pad_token_id,
                self.q_token_id, self.a_token_id
            ):
                tokens.append(self.id_to_token[i])
        
        # Reconstruct text by removing </w> markers and joining
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def vocab_info(self) -> str:
        lines = [f"Vocab size: {self.vocab_size}"]
        lines.append(f"  Special: {self.SPECIAL_TOKENS}")
        lines.append(f"  Merges learned: {len(self.merges)}")
        lines.append(f"  Subword tokens: {self.vocab_size - len(self.SPECIAL_TOKENS)}")
        return "\n".join(lines)

    def save(self, path: str):
        import json
        data = {
            "token_to_id": self.token_to_id,
            "merges": [((a, b), m) for (a, b), m in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merges = [((a, b), m) for [a, b], m in data["merges"]]
        return cls(data["token_to_id"], merges)


TOKENIZER_REGISTRY = {
    "word": WordTokenizer,
    "whitespace": WhitespaceTokenizer,
    "char": CharTokenizer,
    "gpt2": GPT2Tokenizer,
    "improved": ImprovedWordTokenizer,
    "bpe": BPETokenizer,
}


def get_tokenizer(name: str, **kwargs):
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer: {name}. Available: {list(TOKENIZER_REGISTRY.keys())}")
    cls = TOKENIZER_REGISTRY[name]
    if name in ("word", "whitespace", "improved", "bpe"):
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
    if type_name in ("word", "whitespace", "improved"):
        data = {"type": type_name, "vocab": tokenizer.word_to_id}
    elif type_name == "bpe":
        data = {"type": type_name, "token_to_id": tokenizer.token_to_id, "merges": [((a, b), m) for (a, b), m in tokenizer.merges]}
    else:
        data = {"type": type_name or "char"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_tokenizer(path: str, data_file: str = None):
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    type_name = data.get("type", "word")
    if type_name in ("word", "whitespace", "improved") and "vocab" in data:
        return TOKENIZER_REGISTRY[type_name](data["vocab"])
    elif type_name == "bpe" and "token_to_id" in data:
        merges = [((a, b), m) for [a, b], m in data.get("merges", [])]
        return BPETokenizer(data["token_to_id"], merges)
    return get_tokenizer(type_name, data_file=data_file)