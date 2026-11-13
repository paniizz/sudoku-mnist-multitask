# Sudoku-MNIST Multi-Task Learning

A compact PyTorch CNN that reads a **3×3 grid of MNIST digits** and solves three tasks at once — with under **20,000 parameters**.

| Task | Type | Output |
|------|------|--------|
| **Missing digit** | Classification | Which digit (0–9) is absent from the grid |
| **Sort order** | Classification | Ascending / descending / neither for each row & column |
| **Row & column sums** | Regression | Normalized sums for all 3 rows + 3 columns |

---

## Highlights

- **Tiny model** — depthwise-separable residual CNN with **16,146** params (hard-capped at 20k)
- **True multi-task learning** — shared backbone, three specialized heads, weighted joint loss
- **Strong results** — after 50 epochs: **99.95%** missing-digit accuracy, **99.84%** sort accuracy
- **Cross-platform** — auto-selects CUDA → Apple MPS → CPU
- **Reproducible** — fixed seeds, logged configs, saved best checkpoints

---

## Results

### Test metrics

| Run | Epochs | Missing Acc ↑ | Sorted Acc ↑ | Sum MAE ↓ | Test Loss ↓ |
|-----|--------|---------------|--------------|-----------|-------------|
| Short | 20 | 99.55% | 95.69% | 0.0547 | 0.2623 |
| Full | **50** | **99.95%** | **99.84%** | **0.0374** | **0.0325** |

Best checkpoint for the 50-epoch run was selected at **epoch 46** by validation score  
`(missing_acc + sorted_acc) − sum_mae`.

### Training curves (50 epochs)

<p align="center">
  <img src="results/epoch_50/training_history_loss.png" width="32%" alt="Loss" />
  <img src="results/epoch_50/training_history_acc.png" width="32%" alt="Accuracy" />
  <img src="results/epoch_50/training_history_mae.png" width="32%" alt="Sum MAE" />
</p>

<details>
<summary>20-epoch curves</summary>

<p align="center">
  <img src="results/epoch_20/training_history_loss.png" width="32%" alt="Loss" />
  <img src="results/epoch_20/training_history_acc.png" width="32%" alt="Accuracy" />
  <img src="results/epoch_20/training_history_mae.png" width="32%" alt="Sum MAE" />
</p>

</details>

Pretrained weights and metadata live in [`results/epoch_20/`](results/epoch_20/) and [`results/epoch_50/`](results/epoch_50/).

---

## How the data works

Each sample is an **84×84** image: a 3×3 mosaic of MNIST digits (28×28 each).

1. Shuffle digits `0–9`, place **nine** of them on the grid, leave **one** out → `missing_digit`
2. For every row and column, label sort order:
   - `1` = strictly ascending  
   - `2` = strictly descending  
   - `0` = neither  
3. Compute row/column sums and normalize to `[0, 1]` using the known range `[3, 24]`

| Split | Samples | Seed |
|-------|---------|------|
| Train | 50,000 | 42 |
| Val | 10,000 | 43 |
| Test | 20,000 | 44 |

MNIST is downloaded automatically on first run into `data/`.

---

## Architecture

```
Input  (1 × 84 × 84)
   │
   ▼
Stem Conv 1→16
   │
   ▼
Residual DS Blocks  16 → 24 → 32   (stride-2 downsamples)
   │
   ▼
Global Average Pool  →  32-d embedding
   │
   ├── head_missing  →  10 logits      (missing digit)
   ├── head_sorted   →  6 × 3 logits   (row/col sort classes)
   └── head_sums     →  6 sigmoid      (normalized sums)
```

**Loss**

```
L = 1.0 · CE(missing) + 2.0 · CE(sorted, class-weighted) + 8.0 · SmoothL1(sums)
```

Sorted classes use weights `[1, 4, 4]` because ascending/descending rows are rarer than unsorted ones.

Optimizer: **AdamW** (`lr=1e-3`, `weight_decay=1e-4`), batch size **128**.

---

## Project layout

```
sudoku-mnist-multitask/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py          # SudokuMNIST dataset
│   ├── model.py         # SudokuNet
│   ├── train.py         # training & evaluation loop
│   └── utils.py         # seeding, metrics, helpers
└── results/
    ├── epoch_20/        # checkpoint, metrics, plots
    └── epoch_50/
```

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/paniizz/sudoku-mnist-multitask.git
cd sudoku-mnist-multitask

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train

From the **project root**:

```bash
# 20 epochs (default)
python3 -m src.train

# 50 epochs (matches reported best results)
python3 -m src.train --epochs 50

# Useful flags
python3 -m src.train --epochs 50 --batch_size 128 --lr 1e-3 --device mps
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `128` | Mini-batch size |
| `--lr` | `1e-3` | AdamW learning rate |
| `--device` | auto | `cpu` / `cuda` / `mps` |
| `--root_dir` | `./data` | MNIST download path |
| `--out_dir` | `./outputs` | Checkpoints & plots |
| `--seed` | `42` | Reproducibility seed |

Outputs are written to `outputs/`:

- `best_model.pt` — best validation checkpoint  
- `best_model.json` — metrics + config  
- `training_history_{loss,acc,mae}.png` — curves  

### 3. Load a pretrained checkpoint

```python
import torch
from src.model import SudokuNet

model = SudokuNet()
state = torch.load("results/epoch_50/best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

---

## Requirements

- Python **3.9+**
- PyTorch **2.0+** (CUDA or Apple Silicon MPS recommended)
- See [`requirements.txt`](requirements.txt)
