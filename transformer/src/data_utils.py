import re
from typing import Tuple, List, Dict
import html
import torch

WHITELIST = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"


def clean_text_pair(src: str, tgt: str, min_len: int = 5, max_len: int = 64, len_ratio: float = 1.5) -> Tuple[str, str] or None:
    def clean_sentence(sent: str) -> str:
        sent = sent.lower()
        sent = html.unescape(sent)
        sent = re.sub(r"https?://\S+|www\.\S+", "", sent)  # remove URLs
        sent = re.sub(r"<.*?>", "", sent)  # remove html tags
        sent = ''.join(ch for ch in sent if ch in WHITELIST)
        sent = re.sub(r"\s+", " ", sent).strip()  # replace multiple spaces
        return sent

    src_clean = clean_sentence(src)
    tgt_clean = clean_sentence(tgt)

    if not (min_len <= len(src_clean.split()) <= max_len):
        return None
    if not (min_len <= len(tgt_clean.split()) <= max_len):
        return None

    r = len(src_clean.split()) / max(1, len(tgt_clean.split()))
    if r > len_ratio or (1/r) > len_ratio:
        return None
    return src_clean, tgt_clean


def collate_batch(batch: List[Dict], pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2) -> Dict[str, torch.Tensor]:
    src_batch = []
    tgt_input_batch = []
    tgt_output_batch = []
    
    for item in batch:
        src_ids = item['src_ids']
        tgt_ids = item['tgt_ids']
        
        src_batch.append(src_ids)
        tgt_input_batch.append([bos_idx] + tgt_ids)  # BOS + target
        tgt_output_batch.append(tgt_ids + [eos_idx])  # target + EOS
    
    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_input_batch)
    
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []
    src_masks = []
    tgt_masks = []
    
    for src, tgt_in, tgt_out in zip(src_batch, tgt_input_batch, tgt_output_batch):
        src_len = len(src)
        src_padded.append(src + [pad_idx] * (max_src_len - src_len))
        src_masks.append([1] * src_len + [0] * (max_src_len - src_len))
        
        tgt_len = len(tgt_in)
        tgt_input_padded.append(tgt_in + [pad_idx] * (max_tgt_len - tgt_len))
        tgt_output_padded.append(tgt_out + [pad_idx] * (max_tgt_len - tgt_len))
        tgt_masks.append([1] * tgt_len + [0] * (max_tgt_len - tgt_len))
    
    return {
        'src': torch.tensor(src_padded, dtype=torch.long),
        'tgt_input': torch.tensor(tgt_input_padded, dtype=torch.long),
        'tgt_output': torch.tensor(tgt_output_padded, dtype=torch.long),
        'src_mask': torch.tensor(src_masks, dtype=torch.long),
        'tgt_mask': torch.tensor(tgt_masks, dtype=torch.long)
    }

