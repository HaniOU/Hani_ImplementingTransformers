import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer import BPETokenizer

def main():
    corpus = (
        ['hug'] * 10 + 
        ['pug'] * 5 + 
        ['pun'] * 12 + 
        ['bun'] * 4 + 
        ['hugs'] * 5
    )

    vocab_size = 9
    
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size)

    print(f"Final Vocab: {tokenizer.get_vocab()}")

    
    word = 'pugs'
    tokens = tokenizer.encode(word)
    decoded = tokenizer.decode(tokens)
    print(f"'{word}' -> {tokens} -> '{decoded}'")

if __name__ == "__main__":
    main()

