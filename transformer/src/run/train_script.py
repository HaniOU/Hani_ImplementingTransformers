# CUDA_VISIBLE_DEVICES=1 poetry run python train_script.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
import wandb

from modelling.model import TransformerModel
from modelling.scheduler import NoamScheduler, configure_optimizers
from modelling.trainer import Trainer
from data_utils import clean_text_pair, collate_batch, clean_split
from dataset import TranslationDataset
from tokenizer import GPT2BPETokenizer


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")


VOCAB_SIZE = 32000
D_MODEL = 512               
N_HEADS = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 128

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1.0
WARMUP_STEPS = 4000
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

NUM_TRAIN_SAMPLES = 5_900_000
NUM_VAL_SAMPLES = 2900      
MIN_SEQ_LEN = 5
MAX_SEQ_LEN_FILTER = 100

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

CHECKPOINT_PATH = '../checkpoints/best_model.pt'
TOKENIZER_PATH = '../tokenizer_artifacts'

SEED = 42
torch.manual_seed(SEED)


print("Loading WMT17 German-English dataset...")
dataset = load_dataset("wmt17", "de-en")
print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")


train_pairs = clean_split(dataset['train'], NUM_TRAIN_SAMPLES, min_seq_len=MIN_SEQ_LEN, max_seq_len_filter=MAX_SEQ_LEN_FILTER, desc="Cleaning train")
val_pairs = clean_split(dataset['validation'], NUM_VAL_SAMPLES, min_seq_len=MIN_SEQ_LEN, max_seq_len_filter=MAX_SEQ_LEN_FILTER, desc="Cleaning val")

print(f"\nCleaned data: {len(train_pairs)} train, {len(val_pairs)} val")
print(f"\nExample pair:")
print(f"  DE: {train_pairs[0][0]}")
print(f"  EN: {train_pairs[0][1]}")


corpus = [pair[0] for pair in train_pairs] + [pair[1] for pair in train_pairs]

tokenizer = GPT2BPETokenizer()
tokenizer.train(corpus, vocab_size=VOCAB_SIZE)

tokenizer.save(TOKENIZER_PATH + ".json")

print(f"Tokenizer trained with vocab size: {tokenizer.get_vocab_size()}")
print(f"Special tokens: PAD={tokenizer.pad_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}, UNK={tokenizer.unk_id}")

PAD_IDX = tokenizer.pad_id
BOS_IDX = tokenizer.bos_id
EOS_IDX = tokenizer.eos_id

test_text = "ich liebe dich"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"\nTest: '{test_text}'")
print(f"  Encoded IDs: {encoded}")
print(f"  Decoded: '{decoded}'")


train_dataset = TranslationDataset(train_pairs, tokenizer, max_len=MAX_SEQ_LEN)
val_dataset = TranslationDataset(val_pairs, tokenizer, max_len=MAX_SEQ_LEN)

def custom_collate(batch):
    return collate_batch(batch, pad_idx=PAD_IDX, bos_idx=BOS_IDX, eos_idx=EOS_IDX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

batch = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  src: {batch['src'].shape}")
print(f"  tgt_input: {batch['tgt_input'].shape}")
print(f"  tgt_output: {batch['tgt_output'].shape}")

model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LEN,
    use_rope=True,
    use_swiglu=True
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model created on {DEVICE}")
print(f"Total parameters: {num_params:,}")
print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB")

optimizer = configure_optimizers(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.98),
    eps=1e-9
)

scheduler = NoamScheduler(
    optimizer,
    d_model=D_MODEL,
    warmup_steps=WARMUP_STEPS
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

wandb_config = {
    "vocab_size": VOCAB_SIZE,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "num_encoder_layers": NUM_ENCODER_LAYERS,
    "num_decoder_layers": NUM_DECODER_LAYERS,
    "dim_feedforward": DIM_FEEDFORWARD,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "use_rope": True
}

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=DEVICE,
    grad_clip=GRAD_CLIP,
    use_wandb=True,  
    wandb_project="ImplementingTransformers",
    wandb_config=wandb_config
)

print("Training setup complete!")

trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=EPOCHS,
    save_path=CHECKPOINT_PATH,
    best_val_loss=None
)

print("\nTraining complete!")

