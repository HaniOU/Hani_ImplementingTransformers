"""
Main training script for the transformer model on Shakespeare dataset.

Usage:
    poetry run python -m run.main
    
    OR from the src directory:
    python -m run.main
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling.model import TransformerModel
from modelling.trainer import Trainer
from modelling.loss import get_loss_function


# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

# Data Path - Use absolute path based on script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, '../../../input.txt')  # Path to Shakespeare dataset

# Model Architecture
D_MODEL = 128          # Reduced from 256
N_HEADS = 4            # Reduced from 8
NUM_ENCODER_LAYERS = 2 # Reduced from 4
NUM_DECODER_LAYERS = 2 # Reduced from 4
DIM_FEEDFORWARD = 512  # Reduced from 1024
DROPOUT = 0.1
BLOCK_SIZE = 128  # Context length (sequence length)

# Training Parameters
EPOCHS = 3             # Reduced from 10
BATCH_SIZE = 128       # Increased from 64
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

# Loss Function
LOSS_TYPE = 'cross_entropy'  # Options: 'cross_entropy', 'label_smoothing'
LABEL_SMOOTHING = 0.1  # Only used if LOSS_TYPE is 'label_smoothing'

# Data
VAL_SPLIT = 0.1  # 10% for validation

# Device & Misc
DEVICE = 'auto'  # Options: 'auto', 'cuda', 'cpu', 'mps'
SAVE_PATH = 'checkpoints/best_model.pt'
SEED = 42

# Generation
GENERATE_EVERY_N_EPOCHS = 1  # Generate sample text every N epochs
MAX_GENERATE_LENGTH = 200

# ============================================================================
# Character-level Tokenizer
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars[:50])}...")
    
    def encode(self, text: str) -> list:
        """Encode text to token ids."""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, ids: list) -> str:
        """Decode token ids to text."""
        return ''.join([self.idx_to_char[i] for i in ids if i in self.idx_to_char])


# ============================================================================
# Language Model Dataset
# ============================================================================

class ShakespeareDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        Args:
            data: Tensor of token ids
            block_size: Context length
        """
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of block_size + 1 tokens
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input: all except last
        y = chunk[1:]   # Target: all except first (shifted by 1)
        return {'input': x, 'target': y}


def collate_lm_batch(batch):
    """Collate function for language modeling."""
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    return {
        'src': inputs,           # Used as both src and tgt_input for LM
        'tgt_input': inputs,     # Same as src for autoregressive LM
        'tgt_output': targets,   # Shifted targets
        'src_mask': None,
        'tgt_mask': None
    }


# ============================================================================
# Helper Functions
# ============================================================================

def get_device(device_arg: str) -> str:
    """Determine the device to use."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def load_shakespeare(path: str) -> str:
    """Load the Shakespeare dataset."""
    # Handle relative path from the run directory
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def generate_text(model, tokenizer, device, prompt="ROMEO:", max_length=200, temperature=0.8):
    """Generate text from the model."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            # Use the input as both src and tgt for the encoder-decoder model
            output = model(input_tensor, input_tensor)
            
            # Get the last token prediction
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            # Optional: stop at certain length to prevent memory issues
            if input_tensor.size(1) > BLOCK_SIZE:
                input_tensor = input_tensor[:, -BLOCK_SIZE:]
    
    # Decode
    generated_ids = input_tensor[0].tolist()
    return tokenizer.decode(generated_ids)


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function."""
    
    # Set random seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Determine device
    device = get_device(DEVICE)
    print(f"Using device: {device}")
    
    # Load Shakespeare dataset
    print("\n" + "="*50)
    print("Loading Shakespeare Dataset")
    print("="*50)
    
    text = load_shakespeare(DATA_PATH)
    print(f"Total characters: {len(text):,}")
    print(f"First 100 chars: {text[:100]!r}")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    VOCAB_SIZE = tokenizer.vocab_size
    
    # Encode the entire text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Total tokens: {len(data):,}")
    
    # Split into train/val
    n = int(len(data) * (1 - VAL_SPLIT))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # Create datasets
    train_dataset = ShakespeareDataset(train_data, BLOCK_SIZE)
    val_dataset = ShakespeareDataset(val_data, BLOCK_SIZE)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_lm_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_lm_batch,
        num_workers=0
    )
    
    # Create model
    print("\n" + "="*50)
    print("Creating Transformer Model")
    print("="*50)
    model = TransformerModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=BLOCK_SIZE + 100
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Create optimizer and scheduler
    print("\n" + "="*50)
    print("Setting up Optimizer and Scheduler")
    print("="*50)
    
    # Use AdamW with fixed learning rate (simpler and often works well)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99)
    )
    
    # Optional: Use cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * EPOCHS,
        eta_min=LEARNING_RATE / 10
    )
    
    print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"Scheduler: CosineAnnealingLR")
    
    # Create loss function
    criterion = get_loss_function(
        loss_type=LOSS_TYPE,
        smoothing=LABEL_SMOOTHING
    )
    print(f"Loss: {LOSS_TYPE}" + (f" (smoothing={LABEL_SMOOTHING})" if LOSS_TYPE == 'label_smoothing' else ""))
    
    # Move model to device
    model = model.to(device)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        grad_clip=GRAD_CLIP
    )
    
    # Extra data to save with checkpoint
    save_extra = {
        'tokenizer_char_to_idx': tokenizer.char_to_idx,
        'config': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'num_encoder_layers': NUM_ENCODER_LAYERS,
            'num_decoder_layers': NUM_DECODER_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT,
            'block_size': BLOCK_SIZE
        }
    }
    
    # Epoch end callback for text generation
    def on_epoch_end(epoch, train_loss, val_loss):
        if epoch % GENERATE_EVERY_N_EPOCHS == 0:
            print("\n--- Generated Sample ---")
            sample = generate_text(
                model, tokenizer, device, 
                prompt="ROMEO:\n", 
                max_length=MAX_GENERATE_LENGTH,
                temperature=0.8
            )
            print(sample)
            print("------------------------\n")
    
    # Training
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    best_val_loss = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=EPOCHS,
        save_path=SAVE_PATH,
        save_extra=save_extra,
        on_epoch_end=on_epoch_end
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    
    # Final generation
    print("\n--- Final Generated Sample ---")
    sample = generate_text(
        model, tokenizer, device,
        prompt="First Citizen:\n",
        max_length=500,
        temperature=0.8
    )
    print(sample)


if __name__ == "__main__":
    main()
