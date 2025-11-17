import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer import BPETokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_custom_bpe(corpus):
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=64)
    return tokenizer, corpus


def train_huggingface_bpe(corpus):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=295,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
    )
    
    tokenizer.train_from_iterator(corpus, trainer)
    
    return tokenizer


def compare_tokenizers():
    corpus = [
        "Machine learning helps in understanding complex patterns.",
        "Learning machine languages can be complex yet rewarding.",
        "Natural language processing unlocks valuable insights from data.",
        "Processing language naturally is a valuable skill in machine learning.",
        "Understanding natural language is crucial in machine learning."
    ]
    
    custom_tokenizer, _ = train_custom_bpe(corpus)
    
    hf_tokenizer = train_huggingface_bpe(corpus)    
    #print(hf_tokenizer.get_vocab_size())

    print()
    test_sentence = "Machine learning is a subset of artificial intelligence."
    print(f"Test Sentence: {test_sentence}")

    custom_tokens = custom_tokenizer.encode(test_sentence)
    custom_decoded = custom_tokenizer.decode(custom_tokens)
    print()
    print("Custom BPE Tokenization:")

    print()
    print(f"Tokens: {custom_tokens}")
    print(f"Decoded: {custom_decoded}")
    print()

    print()
    print("Huggingface BPE Tokenization:")
    print()
    hf_encoding = hf_tokenizer.encode(test_sentence)
    hf_tokens = hf_encoding.tokens
    hf_ids = hf_encoding.ids
    hf_decoded = hf_tokenizer.decode(hf_ids)
    print(f"Tokens: {hf_tokens}")
    print(f"Decoded: {hf_decoded}")
    print()

if __name__ == "__main__":
    compare_tokenizers()

