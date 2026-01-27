from datasets import load_dataset
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(tgt)
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}

