import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer import BPETokenizer


def main():
    corpus = [
        "Machine learning helps in understanding complex patterns.",
        "Learning machine languages can be complex yet rewarding.",
        "Natural language processing unlocks valuable insights from data.",
        "Processing language naturally is a valuable skill in machine learning.",
        "Understanding natural language is crucial in machine learning."
    ]
    
    vocab_size = 64

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size)

    print(f"Final Vocab: {tokenizer.get_vocab()}")
    
   
    test_sentence = "Machine learning is a subset of artificial intelligence."
    tokens = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(tokens)

    print()
    print(f"Input: \"{test_sentence}\"")
    print()
    print(f"Tokens: {tokens}")
    print()
    print(f"Decoded: {decoded}")
    print()

if __name__ == "__main__":
    main()

