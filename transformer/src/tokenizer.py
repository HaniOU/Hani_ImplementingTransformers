import re
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


class GPT2BPETokenizer:    
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self):
        self.tokenizer = None
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
    
    def train(self, corpus: List[str], vocab_size: int) -> None:
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN],
            min_frequency=2
        )
        
        self.tokenizer.train_from_iterator(corpus, trainer)
        
        self.pad_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.bos_id = self.tokenizer.token_to_id(self.BOS_TOKEN)
        self.eos_id = self.tokenizer.token_to_id(self.EOS_TOKEN)
        self.unk_id = self.tokenizer.token_to_id(self.UNK_TOKEN)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special_ids]
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def save(self, path: str) -> None:
        self.tokenizer.save(path)
    
    @classmethod
    def load(cls, path: str) -> "GPT2BPETokenizer":
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        instance.pad_id = instance.tokenizer.token_to_id(cls.PAD_TOKEN)
        instance.bos_id = instance.tokenizer.token_to_id(cls.BOS_TOKEN)
        instance.eos_id = instance.tokenizer.token_to_id(cls.EOS_TOKEN)
        instance.unk_id = instance.tokenizer.token_to_id(cls.UNK_TOKEN)
        return instance


class BPETokenizer:
    
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self):
        self.vocab = []  
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
    def train(self, corpus: List[str], vocab_size: int) -> None:
        word_freqs = self._get_word_frequencies(corpus)
                
        splits = {word: list(word) for word in word_freqs.keys()}
        
        base_vocab = self._get_base_vocab(splits)
        
        self.vocab = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.vocab.extend(base_vocab)
        
        num_merges = vocab_size - len(self.vocab) 
        
        for i in range(num_merges):
            pair_freqs = self._compute_pair_frequencies(splits, word_freqs)
            
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            splits = self._merge_pair(best_pair, splits)
            
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab.append(new_token)
        
        self._build_mappings()
    
    def _build_mappings(self):
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
    
    def encode(self, text: str) -> List[int]:
        words = self._preprocess_text(text)
        
        ids = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                token_id = self.token_to_id.get(token, self.unk_id)
                ids.append(token_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)
        
        return ''.join(tokens)
    
    def encode_tokens(self, text: str) -> List[str]:
        words = self._preprocess_text(text)
        
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def get_vocab(self) -> List[str]:
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_merges(self) -> Dict[Tuple[str, str], str]:
        return self.merges.copy()
    
    
    def _get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        word_freqs = Counter()
        
        for sentence in corpus:
            words = self._preprocess_text(sentence)
            word_freqs.update(words)
        
        return dict(word_freqs)
    
    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words
    
    def _get_base_vocab(self, splits: Dict[str, List[str]]) -> List[str]:
        vocab = set()
        for word_chars in splits.values():
            vocab.update(word_chars)
        return sorted(list(vocab))
    
    def _compute_pair_frequencies(
        self, 
        splits: Dict[str, List[str]], 
        word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return dict(pair_freqs)
    
    def _merge_pair(
        self, 
        pair: Tuple[str, str], 
        splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        new_splits = {}
        new_token = ''.join(pair)
        
        for word, split in splits.items():
            new_split = []
            i = 0
            
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(new_token)
                    i += 2  
                else:
                    new_split.append(split[i])
                    i += 1
            
            new_splits[word] = new_split
        
        return new_splits
    
    def _tokenize_word(self, word: str) -> List[str]:
        tokens = list(word)
        
        for pair, new_token in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    tokens = tokens[:i] + [new_token] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens