from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=None):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(tgt)
        
        if self.max_len:
            src_ids = src_ids[:self.max_len]
            tgt_ids = tgt_ids[:self.max_len]
        
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}

