# Transformer Project Structure

This project follows a modular approach for building and training transformer models.

## Directory Structure

```
transformer/
├── src/
│   └── transformer/
│       ├── __init__.py
│       ├── modelling/              # Model architecture and training
│       │   ├── __init__.py
│       │   ├── model.py           # Transformer model architecture
│       │   ├── trainer.py         # Training logic
│       │   ├── loss.py            # Loss functions
│       │   └── scheduler.py       # Learning rate schedulers
│       ├── dataset.py              # Data loading and preprocessing
│       └── run/                    # Execution scripts
│           ├── __init__.py
│           └── main.py            # Main training/evaluation script
├── tests/                          # Test modules
│   ├── __init__.py
│   ├── test_model.py
│   └── test_dataset.py
├── pyproject.toml                  # Project dependencies
├── poetry.lock
└── README.md
```

## Module Descriptions

### `modelling/` Module

Contains all model-related code:

- **`model.py`**: Transformer model architecture
  - `TransformerModel`: Main transformer implementation with encoder-decoder architecture
  - Positional encoding
  - Embedding layers

- **`trainer.py`**: Training and evaluation logic
  - `Trainer`: Class for managing training loops, validation, and checkpointing
  - Gradient clipping support
  - Automatic checkpoint saving

- **`loss.py`**: Loss functions
  - `LabelSmoothingCrossEntropy`: Cross-entropy with label smoothing
  - `FocalLoss`: Focal loss for handling class imbalance
  - `get_loss_function()`: Factory function for creating loss functions

- **`scheduler.py`**: Learning rate schedulers
  - `WarmupScheduler`: Linear warmup with decay
  - `CosineAnnealingWarmup`: Cosine annealing with warmup
  - `NoamScheduler`: Original transformer scheduler from "Attention is All You Need"
  - `get_scheduler()`: Factory function for creating schedulers

### `dataset.py` Script

Contains data loading and preprocessing:
- `load_and_prepare_data()`: Load datasets from Hugging Face
- `clean_data()`: Data cleaning logic
- `prepare_data()`: Data preparation (tokenization, padding, etc.)

### `run/main.py` Script

Main entry point for training and evaluation:
- Command-line argument parsing
- Training pipeline orchestration
- Evaluation logic
- Checkpoint management

### `tests/` Directory

Contains test modules for all components:
- `test_model.py`: Tests for model architecture
- `test_dataset.py`: Tests for data loading and preprocessing

## Usage

### Training a Model

```bash
cd transformer
python -m transformer.run.main \
    --dataset wikitext \
    --batch-size 32 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --output-dir ./checkpoints
```

### Running Tests

```bash
cd transformer
poetry run pytest tests/
```

### Using Components in Code

```python
from transformer.modelling import TransformerModel, Trainer, get_loss_function, get_scheduler
from transformer.dataset import load_and_prepare_data

# Load data
dataset = load_and_prepare_data("wikitext", split="train")

# Create model
model = TransformerModel(
    vocab_size=30000,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = get_loss_function("label_smoothing", smoothing=0.1)
scheduler = get_scheduler("noam", optimizer, d_model=512, warmup_steps=4000)

# Train
trainer = Trainer(model, optimizer, criterion, scheduler=scheduler)
trainer.train(train_loader, val_loader, num_epochs=10)
```

## Dependencies

Key dependencies managed by Poetry:
- `torch`: PyTorch deep learning framework
- `torchvision`: Vision utilities
- `torchaudio`: Audio processing
- `datasets`: Hugging Face datasets library
- `pytest`: Testing framework (dev dependency)

## Development

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Activate virtual environment:
   ```bash
   poetry shell
   ```

3. Run tests:
   ```bash
   poetry run pytest
   ```

4. Add new dependencies:
   ```bash
   poetry add package-name
   ```

