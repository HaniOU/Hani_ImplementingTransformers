# Transformer Model - Vollständige Dokumentation

## Projektübersicht

Dieses Projekt implementiert einen **Encoder-Decoder Transformer** von Grund auf für die **Deutsch → Englisch Übersetzung** unter Verwendung des **WMT17** Datensatzes. Das Projekt folgt der originalen "Attention Is All You Need" (Vaswani et al., 2017) Architektur mit modernen Erweiterungen wie Rotary Position Embeddings (RoPE).

### Projektstruktur

```
transformer/
├── src/
│   ├── modelling/
│   │   ├── model.py           # TransformerModel (Encoder-Decoder)
│   │   ├── attention.py       # Attention, MultiHeadAttention
│   │   ├── positional_encoding.py  # Sinusoidal PE, RoPE
│   │   ├── functional.py      # Encoder/Decoder Layers
│   │   ├── scheduler.py       # NoamScheduler, WarmupScheduler
│   │   ├── loss.py            # LabelSmoothingCrossEntropy
│   │   └── trainer.py         # Training Loop mit WandB
│   ├── tokenizer.py           # BPETokenizer, GPT2BPETokenizer
│   ├── dataset.py             # TranslationDataset
│   ├── data_utils.py          # Datenreinigung, Collate-Funktion
│   └── run/
│       └── main.ipynb         # Hauptskript für Training/Evaluation
└── practicals/                # Praktikums-Notebooks
```

---

## 1. Transformer Architektur

### 1.1 Grundstruktur

Der Transformer besteht aus zwei Hauptkomponenten:

```
Input (Deutsch) ──► [Encoder] ──► Memory ──┐
                                           │
                                           ▼
Output (Englisch) ◄── [Decoder] ◄──────────┘
```

**Encoder:**
- Verarbeitet die gesamte Eingabesequenz (deutscher Satz)
- Erzeugt kontextuelle Repräsentationen für jedes Token
- Besteht aus N identischen Schichten

**Decoder:**
- Generiert die Ausgabesequenz Token für Token (autoregressiv)
- Nutzt Self-Attention mit Future Masking
- Nutzt Cross-Attention zu den Encoder-Outputs
- Besteht aus N identischen Schichten

### 1.2 Modell-Parameter

| Parameter | Beschreibung | Standard | Medium Config |
|-----------|--------------|----------|---------------|
| `vocab_size` | Vokabulargröße | - | 16,000 |
| `d_model` | Embedding-Dimension | 512 | 256 |
| `n_heads` | Anzahl Attention Heads | 8 | 8 |
| `num_encoder_layers` | Encoder-Schichten | 6 | 4 |
| `num_decoder_layers` | Decoder-Schichten | 6 | 4 |
| `dim_feedforward` | FFN innere Dimension | 2048 | 1024 |
| `dropout` | Dropout Rate | 0.1 | 0.1 |
| `max_len` | Max. Sequenzlänge | 5000 | 5000 |

### 1.3 Dimensionsbeziehungen

Die wichtigste Beziehung:
```
d_head = d_model / n_heads
```

Für unser Medium-Modell:
- `d_model = 256`
- `n_heads = 8`  
- `d_head = 256 / 8 = 32`

Die Feed-Forward Dimension ist typischerweise `4 × d_model`:
- `dim_feedforward = 4 × 256 = 1024`

---

## 2. Word Embeddings

### 2.1 Implementierung

```python
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)
```

### 2.2 Mathematische Definition

$$\text{Emb}(\mathbf{x}) = \mathbb{1}(\mathbf{x}) \mathbf{E}$$

wobei:
- $\mathbf{x}$ = Vektor von Token-Indizes
- $\mathbb{1}$ = One-Hot Encoding Funktion
- $\mathbf{E} \in \mathbb{R}^{V \times d_{model}}$ = Embedding-Matrix

Die Embedding-Schicht fungiert als **Lookup-Tabelle**: Jede Zeile der Matrix ist die Embedding-Repräsentation des entsprechenden Tokens.

### 2.3 Parameter Sharing

**Wichtig:** Im Transformer werden die Embedding-Gewichte zwischen drei Stellen geteilt:
1. Source Embedding (Encoder-Input)
2. Target Embedding (Decoder-Input)
3. Pre-Softmax Linear Transformation (Output Projection)

```python
# In TransformerModel.__init__:
self.output_projection.weight = self.embedding.embedding.weight
```

**Vorteile des Parameter Sharing:**
- **Reduzierte Parameteranzahl:** Statt 3 separate Matrizen nur eine
- **Bessere Generalisierung:** Ähnliche Tokens werden ähnlich behandelt
- **Semantische Konsistenz:** Ein Token hat dieselbe "Bedeutung" als Input und Output
- **Regulierung:** Verhindert Overfitting durch weniger Parameter

---

## 3. Positional Encoding

### 3.1 Warum brauchen wir Position Information?

Der Transformer hat keine rekurrente Struktur wie RNNs. Attention behandelt alle Positionen gleichzeitig und hat keine inhärente Vorstellung von Reihenfolge. Ohne Positionsinformation wäre "Der Hund beißt den Mann" identisch zu "Der Mann beißt den Hund".

### 3.2 Sinusoidale Positional Encodings (Original)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

**Mathematische Formel:**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 3.3 Eigenschaften der Positional Encodings (Beweis)

**Eigenschaft 1: Lineare Transformation für festes Offset k**

Für ein festes Offset $k$ kann $PE_{t+k}$ als lineare Funktion von $PE_t$ dargestellt werden:

$$PE_{t+k} = M_k \cdot PE_t$$

wobei $M_k$ eine Rotationsmatrix ist. Dies liegt daran, dass:

$$\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$
$$\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$

Somit kann für jede Dimension:
$$\begin{bmatrix} \sin(\omega(t+k)) \\ \cos(\omega(t+k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix} \begin{bmatrix} \sin(\omega t) \\ \cos(\omega t) \end{bmatrix}$$

**Eigenschaft 2: Geometrische Progression der Wellenlängen**

Die Wellenlängen bilden eine geometrische Reihe von $2\pi$ bis $10000 \cdot 2\pi$:

Für Dimension $2i$ ist die Wellenlänge:
$$\lambda_i = 2\pi \cdot 10000^{2i/d_{model}}$$

- Bei $i = 0$: $\lambda_0 = 2\pi$
- Bei $i = d_{model}/2 - 1$: $\lambda_{max} \approx 10000 \cdot 2\pi$

Dies ermöglicht dem Modell, sowohl kurz- als auch langreichweitige Positionsbeziehungen zu lernen.

---

## 4. Rotary Position Embeddings (RoPE) - Moderne Erweiterung

### 4.1 Motivation

RoPE (Rotary Position Embeddings) ist eine moderne Alternative zu sinusoidalen Positional Encodings, verwendet in:
- LLaMA
- GPT-NeoX
- PaLM

**Vorteile gegenüber sinusoidalen PE:**
- Bessere Extrapolation zu längeren Sequenzen
- Positionsinformation direkt in Attention-Scores integriert
- Relative statt absolute Positionsinformation

### 4.2 Implementierung

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
```

### 4.3 Mathematisches Prinzip

RoPE rotiert Query- und Key-Vektoren basierend auf ihrer Position:

$$f_q(x_m, m) = R_m x_m$$

wobei $R_m$ eine Rotationsmatrix ist:

$$R_m = \begin{bmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & & \\ \sin(m\theta_1) & \cos(m\theta_1) & & \\ & & \ddots & \\ & & & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\ & & & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2}) \end{bmatrix}$$

Das Skalarprodukt $q_m^T k_n$ enthält dann nur den **relativen** Abstand $(m-n)$:

$$q_m^T k_n = x_m^T R_{m-n} x_n$$

### 4.4 Verwendung im Projekt

```python
# In TransformerModel:
model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    use_rope=True,  # Aktiviert RoPE statt sinusoidaler PE
    ...
)
```

**Wichtig:** RoPE wird nur für Self-Attention verwendet, nicht für Cross-Attention im Decoder.

---

## 5. Attention Mechanismus

### 5.1 Scaled Dot-Product Attention

```python
class Attention(nn.Module):
    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future
        
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if self.mask_future:
            seq_len = query.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        return output
```

**Formel:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Warum $\sqrt{d_k}$?**
- Ohne Skalierung wachsen die Dot-Products mit höherer Dimension
- Große Werte führen zu extremen Softmax-Outputs (nahe 0 oder 1)
- Dies verursacht verschwindende Gradienten
- Skalierung stabilisiert das Training

### 5.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, mask_future=False, use_rope=False, max_seq_len=5000):
        super().__init__()
        self.head_dim = embedding_dim // num_heads
        
        self.query_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
```

**Formel:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

wobei:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Vorteile von Multi-Head Attention:**
- Verschiedene Heads können verschiedene Aspekte lernen (Syntax, Semantik, Koreferenz)
- Parallele Berechnung möglich
- Größere Repräsentationskapazität

### 5.3 Masking

**Padding Mask:**
- Verhindert, dass `<PAD>` Tokens zur Attention beitragen
- Binäre Matrix: 1 = echtes Token, 0 = Padding
- Padding darf die "Bedeutung" nicht verändern

**Future Mask (Causal Mask):**
- Verhindert, dass der Decoder "in die Zukunft schaut"
- Lower-Triangular Matrix
- Essentiell für autoregressives Training

```python
# Future Mask Beispiel für Sequenzlänge 4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

---

## 6. Transformer Layers

### 6.1 Encoder Layer

```python
class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1, use_rope=False, max_seq_len=5000):
        self.self_attention = MultiHeadAttention(...)
        self.feature_transformation = PositionWiseFeedForward(...)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        # Sublayer 1: Self-Attention + Residual + LayerNorm
        attn_output = self.self_attention(x, x, x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm_1(x + attn_output)
        
        # Sublayer 2: FFN + Residual + LayerNorm
        ff_output = self.feature_transformation(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm_2(x + ff_output)
        
        return x
```

### 6.2 Decoder Layer

Der Decoder hat **drei** Sublayer:
1. **Masked Self-Attention** (mit Future Mask)
2. **Cross-Attention** (Query vom Decoder, Key/Value vom Encoder)
3. **Feed-Forward Network**

```python
class TransformerDecoderLayer(nn.Module):
    def forward(self, x, encoder, encoder_attention_mask=None, attention_mask=None):
        # Sublayer 1: Masked Self-Attention
        self_attn = self.self_attention(x, x, x, attention_mask)
        x = self.layer_norm_1(x + self.dropout(self_attn))
        
        # Sublayer 2: Cross-Attention (Q: decoder, K/V: encoder)
        cross_attn = self.encoder_attention(x, encoder, encoder, encoder_attention_mask)
        x = self.layer_norm_2(x + self.dropout(cross_attn))
        
        # Sublayer 3: Feed-Forward
        ff = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout(ff))
        
        return x
```

### 6.3 Position-Wise Feed-Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, feature_dim):
        self.linear1 = nn.Linear(input_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, input_dim)
        
    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))
```

**Funktion:**
- Transformiert jede Position unabhängig
- Erhöht die Modellkapazität (4× Expansion)
- Nicht-Linearität durch ReLU

### 6.4 Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

wobei:
- $\mu$ = Mittelwert über die Feature-Dimension
- $\sigma$ = Standardabweichung über die Feature-Dimension
- $\gamma, \beta$ = Lernbare Parameter

**Funktion:**
- Stabilisiert das Training
- Verhindert Internal Covariate Shift
- Ermöglicht höhere Learning Rates

### 6.5 Residual Connections

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Funktion:**
- Verhindert Informationsverlust in tiefen Netzwerken
- Mildert das Vanishing Gradient Problem
- Ermöglicht das Training sehr tiefer Netzwerke

---

## 7. Tokenizer

### 7.1 Byte-Pair Encoding (BPE) Algorithmus

**Schritt 1: Basis-Vokabular**
- Alle einzigartigen Zeichen im Korpus

**Schritt 2: Merge-Regeln lernen**
```
Wortfrequenzen: {'hug': 10, 'pug': 5, 'pun': 12, 'bun': 4, 'hugs': 5}

Iteration 1:
  Co-occurrence: {('u','g'): 20, ('h','u'): 15, ('p','u'): 17, ...}
  Merge: ('u','g') → 'ug'
  
Iteration 2:
  Co-occurrence: {('u','n'): 16, ('h','ug'): 15, ...}
  Merge: ('u','n') → 'un'
```

**Schritt 3: Encoding**
```
"pugs" → ['p','u','g','s'] → ['p','ug','s']
```

### 7.2 Implementierungen

**BPETokenizer (Eigene Implementierung):**
- Langsam aber lehrreich
- Vollständige Kontrolle

**GPT2BPETokenizer (HuggingFace Wrapper):**
- Schnell (C++ Backend)
- Produktionsreif

```python
class GPT2BPETokenizer:    
    PAD_TOKEN = "<pad>"  # ID: 0
    BOS_TOKEN = "<s>"    # ID: 1
    EOS_TOKEN = "</s>"   # ID: 2
    UNK_TOKEN = "<unk>"  # ID: 3
    
    def train(self, corpus, vocab_size):
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN],
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(corpus, trainer)
```

### 7.3 Special Tokens

| Token | ID | Funktion |
|-------|-----|----------|
| `<pad>` | 0 | Padding für gleichlange Batches |
| `<s>` | 1 | Begin-of-Sequence (Decoder-Start) |
| `</s>` | 2 | End-of-Sequence (Generierung stoppen) |
| `<unk>` | 3 | Unbekannte Tokens |

---

## 8. Datenverarbeitung

### 8.1 Datenreinigung

```python
WHITELIST = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"

def clean_text_pair(src, tgt, min_len=5, max_len=64, len_ratio=1.5):
    # 1. Lowercase
    # 2. HTML unescape
    # 3. URLs entfernen
    # 4. HTML-Tags entfernen
    # 5. Nur Whitelist-Zeichen behalten
    # 6. Längenfilter
    # 7. Ratio-Filter (src/tgt Länge)
```

### 8.2 Dataset

```python
class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=None):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(tgt)
        
        if self.max_len:
            src_ids = src_ids[:self.max_len]
            tgt_ids = tgt_ids[:self.max_len]
        
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
```

### 8.3 Collate Funktion (Batching)

```python
def collate_batch(batch, pad_idx=0, bos_idx=1, eos_idx=2):
    # Input:  src_ids, tgt_ids
    # Output:
    #   src:        [src_ids... PAD PAD]
    #   tgt_input:  [BOS tgt_ids... PAD PAD]
    #   tgt_output: [tgt_ids... EOS PAD PAD]
    #   src_mask:   [1 1 1 ... 0 0]
    #   tgt_mask:   [1 1 1 ... 0 0]
```

**Warum BOS/EOS?**
- **BOS** (`<s>`): Signalisiert dem Decoder den Start der Generierung
- **EOS** (`</s>`): Trainingsziel - Modell lernt, wann es aufhören soll

---

## 9. Loss Funktion

### 9.1 Cross-Entropy Loss

$$\mathcal{L}_{CE} = -\sum_{i} y_i \log(\hat{y}_i)$$

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

`ignore_index=PAD_IDX` stellt sicher, dass Padding-Tokens nicht zur Loss-Berechnung beitragen.

### 9.2 Label Smoothing Cross-Entropy

**Problem mit hartem One-Hot:**
- Fördert Overconfidence
- Schlechte Generalisierung
- Modell wird "zu sicher"

**Label Smoothing Lösung:**

$$y_i^{smooth} = \begin{cases} 
1 - \epsilon & \text{wenn } i = \text{target} \\
\frac{\epsilon}{V-1} & \text{sonst}
\end{cases}$$

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        pred_log_prob = F.log_softmax(pred, dim=-1)
        loss = -(true_dist * pred_log_prob).sum(dim=-1)
        return loss.mean()
```

**Verwendung:**
```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
```

---

## 10. Optimizer und Learning Rate Scheduler

### 10.1 AdamW Optimizer

**Update-Gleichungen:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Warum Bias Correction?**
- $m_0 = 0$ und $v_0 = 0$ (Initialisierung)
- In frühen Steps sind $m_t$ und $v_t$ zu klein
- Bias Correction kompensiert diesen Bias

**Decoupled Weight Decay:**
$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

In AdamW wird Weight Decay **separat** angewandt (nicht in den Gradienten):
```python
# Kein Weight Decay für bias und LayerNorm:
if 'bias' in name or 'layer_norm' in name:
    no_decay_params.append(param)
else:
    decay_params.append(param)
```

### 10.2 Noam Learning Rate Scheduler

```python
class NoamScheduler(_LRScheduler):
    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.d_model ** (-0.5)
        lr = scale * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [lr * self.factor]
```

**Formel:**
$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})$$

**Verhalten:**
1. **Warmup Phase** (step < warmup_steps): Lineare Erhöhung
2. **Decay Phase** (step > warmup_steps): Inverse Square Root Decay

```
LR
│     ╱╲
│    ╱  ╲
│   ╱    ╲_____
│  ╱           ╲____
│ ╱                 ╲___
└──────────────────────── Step
  ^warmup
```

---

## 11. Training

### 11.1 Training Loop

```python
class Trainer:
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        for batch in dataloader:
            # 1. Forward Pass
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # 2. Loss Berechnung
            loss = self.criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # 3. Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 4. Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # 5. Parameter Update
            self.optimizer.step()
            
            # 6. Learning Rate Update
            self.scheduler.step()
```

### 11.2 Teacher Forcing

Während des Trainings bekommt der Decoder die **echten** vorherigen Tokens (nicht seine eigenen Vorhersagen):

```
Training:
  Input:  [BOS, "Der", "Hund"]
  Target: ["Der", "Hund", EOS]
  
  Der Decoder sieht IMMER die richtigen Tokens, auch wenn er falsch vorhersagt.
```

### 11.3 Weights & Biases Integration

```python
if self.use_wandb:
    wandb.init(project=wandb_project, config=wandb_config)
    wandb.watch(model, log="all", log_freq=100)
    
# Während Training:
wandb.log({
    "train/loss": loss.item(),
    "train/learning_rate": current_lr,
    "train/step": self.step
})

# Nach Validation:
wandb.log({"val/loss": avg_loss, "val/epoch": epoch})
```

---

## 12. Inferenz (Autoregressive Generation)

### 12.1 Greedy Decoding

```python
def generate(self, src, src_mask=None, bos_idx=1, eos_idx=2, max_length=100):
    # 1. Encode source
    encoder_output = self.encode(src, src_mask)
    
    # 2. Start mit BOS token
    generated = torch.full((batch_size, 1), bos_idx)
    
    # 3. Autoregressive Generierung
    for _ in range(max_length - 1):
        # Decode aktuell generierte Sequenz
        decoder_output = self.decode(generated, encoder_output, ...)
        
        # Nehme letztes Token
        logits = self.output_projection(decoder_output[:, -1, :])
        
        # Wähle Token mit höchster Wahrscheinlichkeit
        next_token = logits.argmax(dim=-1, keepdim=True)
        
        # Anhängen
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stoppen wenn EOS
        if (next_token == eos_idx).all():
            break
    
    return generated
```

### 12.2 Translate Funktion

```python
def translate(model, src_text, tokenizer, device, max_length=100):
    model.eval()
    
    # Tokenize input
    src_ids = tokenizer.encode(src_text)
    src_tensor = torch.tensor([src_ids]).to(device)
    
    # Generate
    generated = model.generate(src_tensor, bos_idx=tokenizer.bos_id, eos_idx=tokenizer.eos_id)
    
    # Decode output
    translation = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    return translation
```

---

## 13. Evaluation

### 13.1 BLEU Score

**BLEU** (Bilingual Evaluation Understudy) misst die Übereinstimmung von n-Grammen zwischen Vorhersage und Referenz:

$$BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

wobei:
- $p_n$ = Precision für n-Gramme
- $BP$ = Brevity Penalty (bestraft zu kurze Übersetzungen)
- $w_n$ = Gewichte (typisch: $1/N$)

```python
from evaluate import load
bleu_metric = load("bleu")

predictions = [translate(model, src, tokenizer, device) for src, _ in test_pairs]
references = [[ref] for _, ref in test_pairs]

bleu_score = bleu_metric.compute(predictions=predictions, references=references)
```

---

## 14. Zusammenfassung der Theoretischen Fragen

### Practical 1

**Frage 6: Warum brauchen wir Positionsinformation?**

Der Attention-Mechanismus ist permutationsinvariant - er behandelt alle Tokens gleichzeitig ohne Reihenfolgeninformation. Ohne Positionsencoding wäre "Hund beißt Mann" = "Mann beißt Hund". Die Position jedes Tokens muss explizit encodiert werden, um Satzstruktur und Grammatik zu erfassen.

### Practical 2

**Frage 1: Rolle der zwei Masks**

1. **Padding Mask:** Verhindert, dass Padding-Tokens (`<PAD>`) zur Attention beitragen. Ohne diese Maske würde das Modell "lernen", dass alle Sätze gleich lang sind.

2. **Future Mask (Causal Mask):** Verhindert im Decoder, dass Tokens auf zukünftige Tokens zugreifen. Essentiell für autoregressives Training - das Modell darf nicht "schummeln".

### Practical 4

**Frage 4: Eigenschaften der Positional Encodings (Beweise)**

Siehe Abschnitt 3.3 für vollständige Beweise:
- PE für Position $t+k$ ist lineare Transformation von PE für Position $t$
- Wellenlängen bilden geometrische Progression von $2\pi$ bis $10000 \cdot 2\pi$

### Practical 5

**Frage 1: Position-Wise Feed-Forward Layer**

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2 = ReLU(xW_1 + b_1)W_2 + b_2$$

Die FFN-Schicht:
- Wird auf jede Position unabhängig angewandt (daher "position-wise")
- Erhöht die Modellkapazität durch Dimensionserweiterung (typisch 4×)
- Fügt Nicht-Linearität hinzu (ReLU)
- Erlaubt komplexere Transformationen als nur Attention

**Frage 4: Layer Normalization**

$$LayerNorm(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Funktion:
- Normalisiert Aktivierungen innerhalb jeder Schicht
- Verhindert Internal Covariate Shift
- Stabilisiert Gradienten
- Ermöglicht höhere Learning Rates
- Beschleunigt Konvergenz

### Practical 7

**Frage 1: Transformer-Struktur**

Siehe Abschnitt 1 für vollständige Struktur:
- Encoder: Embedding → PE → N × (Self-Attention → FFN)
- Decoder: Embedding → PE → N × (Masked Self-Attention → Cross-Attention → FFN)
- Output: Linear Projection → Softmax

**Frage 2: Parameter Sharing Vorteile**

Siehe Abschnitt 2.3:
- Reduzierte Parameteranzahl
- Bessere Generalisierung
- Semantische Konsistenz
- Implizite Regularisierung

### Practical 8

**Frage 1: Noam Learning Rate Scheduler**

Siehe Abschnitt 10.2:
$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})$$

Warmup verhindert instabiles Training zu Beginn, Decay verhindert Oszillation später.

**Frage 3: AdamW - Bias Correction und Decoupled Weight Decay**

Siehe Abschnitt 10.1:
- **Bias Correction:** Kompensiert die Initialisierung von $m_0 = v_0 = 0$
- **Decoupled Weight Decay:** Trennt Regularisierung von Gradientenberechnung für bessere Konvergenz

---

## 15. Moderne Erweiterungen (Was wir hinzugefügt haben)

### 15.1 Rotary Position Embeddings (RoPE)
- Ersetzt sinusoidale Positional Encodings
- Relative statt absolute Positionsinformation
- Aktiviert mit `use_rope=True`

### 15.2 Label Smoothing
- Verhindert Overconfidence
- Bessere Generalisierung
- Aktiviert mit `label_smoothing=0.1` in CrossEntropyLoss

### 15.3 Weights & Biases Integration
- Experiment Tracking
- Loss/Learning Rate Visualisierung
- Model Watching
- Aktiviert mit `use_wandb=True` im Trainer

---

## 16. Konfiguration für Training

```python
# Model Architecture
D_MODEL = 256
N_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 4000
GRAD_CLIP = 1.0

# Data
VOCAB_SIZE = 16000
MAX_LEN = 128
NUM_TRAIN_SAMPLES = 500000
NUM_VAL_SAMPLES = 20000
NUM_TEST_SAMPLES = 5000

# Modern Improvements
USE_ROPE = True
LABEL_SMOOTHING = 0.1
USE_WANDB = True
```

---

*Dokumentation erstellt für das Projekt "Implementing Transformers" - Data Science Master, 2025/2026*
