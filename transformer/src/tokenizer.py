import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import os
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


class BPETokenizer:
    def __init__(self):
        self.vocab = []  
        self.merges = {} 
        
    def train(self, corpus: List[str], vocab_size: int) -> None: # corpus = list of sentences
        word_freqs = self._get_word_frequencies(corpus)
                
        splits = {word: list(word) for word in word_freqs.keys()} # {hug: ['h', 'u', 'g']}
        
        self.vocab = self._get_base_vocab(splits) # ['h', 'u', 'g']
        
        num_merges = vocab_size - len(self.vocab) 
        
        for i in range(num_merges):
            pair_freqs = self._compute_pair_frequencies(splits, word_freqs) # {('h', 'u'): 15, ('u', 'g'): 20}
            
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get) # ('u', 'g')
            
            splits = self._merge_pair(best_pair, splits) # {hug: ['h', 'ug']}
            
            new_token = ''.join(best_pair) # 'ug'
            self.merges[best_pair] = new_token # {('u', 'g'): 'ug'}
            self.vocab.append(new_token) # ['h', 'u', 'g','ug']
    
    def encode(self, text: str) -> List[str]:
        words = self._preprocess_text(text)
        
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def decode(self, tokens: List[str]) -> str:
        return ' '.join(tokens)
    
    def get_vocab(self) -> List[str]:
        return self.vocab.copy()
    
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

class GPT2BPETokenizer:
    def __init__(self, 
                 corpus: Optional[List[str]]=None, 
                 vocab_size: int=50000, 
                 special_tokens: Optional[List[str]]=None,
                 artifact_dir: str="./tokenizer_artifacts"):
        if special_tokens is None:
            special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tk.train_from_iterator(corpus, trainer)
        tk.model.save(artifact_dir)  # Erzeugt vocab.json + merges.txt
        self.tokenizer = GPT2Tokenizer.from_pretrained(artifact_dir, 
                                                        unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    def encode(self, text: str, **kwargs):
        # Gibt token ids zurück
        return self.tokenizer.encode(text, **kwargs)
    def decode(self, ids: List[int], **kwargs):
        return self.tokenizer.decode(ids, **kwargs)
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    def get_vocab_size(self):
        return self.tokenizer.vocab_size