# CUDA_VISIBLE_DEVICES=1 poetry run python inference_script.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

import torch
from modelling.model import TransformerModel
from tokenizer import GPT2BPETokenizer
#import evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from modelling.model import TransformerModel
from modelling.scheduler import NoamScheduler, configure_optimizers
from modelling.trainer import Trainer
from data_utils import clean_text_pair, collate_batch, clean_split, translate
from dataset import TranslationDataset
from tokenizer import GPT2BPETokenizer
import sacrebleu


VOCAB_SIZE = 32000
D_MODEL = 256               
N_HEADS = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 128

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1.0
WARMUP_STEPS = 4000
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

NUM_TRAIN_SAMPLES = 500_000
NUM_VAL_SAMPLES = 2500      
NUM_TEST_SAMPLES = 2500      
MIN_SEQ_LEN = 5
MAX_SEQ_LEN_FILTER = 100

num_examples = 10

CHECKPOINT_PATH = '../checkpoints/model0.pt'
TOKENIZER_PATH = '../tokenizer_artifacts.json'

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")


tokenizer = GPT2BPETokenizer.load(TOKENIZER_PATH)

print(f"Tokenizer geladen mit vocab size: {tokenizer.get_vocab_size()}")
print(f"Special tokens: PAD={tokenizer.pad_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")


print("Loading WMT17 German-English dataset...")
dataset = load_dataset("wmt17", "de-en")
print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")

test_pairs = clean_split(dataset['test'], 2500, min_seq_len=MIN_SEQ_LEN, max_seq_len_filter=MAX_SEQ_LEN_FILTER, desc="Cleaning test")

def custom_collate(batch):
    return collate_batch(batch, pad_idx=tokenizer.pad_id, bos_idx=tokenizer.bos_id, eos_idx=tokenizer.eos_id)

test_dataset = TranslationDataset(test_pairs, tokenizer, max_len=MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)


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
)

print(f"Modell erstellt mit {sum(p.numel() for p in model.parameters()):,} Parametern")


checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

print("Checkpoint enthält:")
for key in checkpoint.keys():
    print(f"  - {key}")

print(f"\nTrainiert für {checkpoint.get('epoch', '?')} Epochen")
print(f"Train Loss: {checkpoint.get('train_loss', '?'):.4f}")
print(f"Val Loss: {checkpoint.get('val_loss', '?'):.4f}")


state_dict = checkpoint['model_state_dict']
skip_keys = {k for k in state_dict if 'rope' in k and ('cos_cached' in k or 'sin_cached' in k)}
state_dict = {k: v for k, v in state_dict.items() if k not in skip_keys}
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()  

print("Gewichte erfolgreich geladen!")


test_sentences = [
    "ich liebe dich",
    "das wetter ist heute schön",
    "wo ist der bahnhof?",
    "ich möchte ein bier bestellen",
]

print("Übersetzungen (Deutsch -> Englisch):")
print("=" * 50)

for sentence in test_sentences:
    print(f"DE: {sentence}")
    translation = translate(model, tokenizer, sentence, device=DEVICE, debug=False)
    print(f"EN: {translation}")
    print("-" * 50)

predictions = []
references = []

model.eval()
for src_text, ref_text in tqdm(test_pairs, desc="Generating translations"):
    pred_text = translate(model, tokenizer, src_text, device=DEVICE, max_len=MAX_SEQ_LEN)
    predictions.append(pred_text)
    references.append(ref_text)  

#bleu = evaluate.load("bleu")
#results = bleu.compute(predictions=predictions, references=references)
bleu = sacrebleu.corpus_bleu(predictions, [references])

print("\n" + "="*50)
print("BLEU SCORE RESULTS")
print("="*50)
print(f"BLEU Score: {bleu.score:.4f}  (= {bleu.score*100:.2f})")
print(f"Precisions: {[f'{p:.4f}' for p in bleu.precisions]}")
print(f"Brevity Penalty: {bleu.bp:.4f}")
print(f"Length Ratio: {bleu.sys_len / bleu.ref_len:.4f}")


print("Example Translations:")
print("="*70)

for i in range(num_examples):
    src, ref = test_pairs[i]
    pred = predictions[i]
    
    ind_bleu = sacrebleu.sentence_bleu(pred, [ref]).score
    
    print(f"\nExample {i+1} (BLEU: {ind_bleu:.4f})")
    print(f"  DE:   {src}")
    print(f"  Pred: {pred}")
    print(f"  Ref:  {ref}")