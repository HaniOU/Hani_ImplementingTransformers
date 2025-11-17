import re
from typing import Tuple
import html

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
