# QM9 Molecular Property Prediction: GCN vs. DimeNet++ vs. EGNN vs. PaiNN
Codebase for the Course Mini-Project — **24-788 Introduction to Deep Learning (Spring 2026)**.

**Task**: Predict the HOMO-LUMO gap (Δε, target index 4) of small organic molecules from the QM9 dataset (~130k molecules), framed as a graph regression problem.

---

## Models

| Directory | Model | Description |
|---|---|---|
| `dimenet/` | DimeNet++ | Directional message-passing GNN using 3D bond lengths and angles |
| `egnn/` | EGNN | E(n)-equivariant GNN using 3D atom positions and pairwise distances |
| `painn/` | PaiNN | Equivariant GNN with coupled scalar and vector message passing |
| `gcn_baseline/` | GCN | Standard graph convolutional network; course baseline |

DimeNet++ and GCN are trained on the following fixed split:
- **Train**: indices 0–109,999 (110k molecules)
- **Val**: indices 110,000–119,999 (10k molecules)
- **Test**: indices 120,000+ (~10.8k molecules)

EGNN and PaiNN are trained on the following split:
- **Train**: indices 0–99,999 (100k molecules)
- **Val**: indices 100,000–117,999 (18k molecules)
- **Test**: indices 118,000+ (~13k molecules)

Target normalization is computed from the training set only.

---

## Repository Structure

```
repo/
├── dimenet/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── requirements.txt
│   ├── train.py
│   ├── dimenet_best_model.pt
│   ├── dimenet_config.json
│   └── reproduce_results.ipynb
├── egnn/
│   ├── requirements.txt
│   ├── train.py
│   ├── egnn_best_model.pt
│   ├── egnn_config.json
│   └── reproduce_results.ipynb
├── painn/
│   ├── requirements.txt
│   ├── train.py
│   ├── painn_best_model.pt
│   ├── painn_config.json
│   └── reproduce_results.ipynb
├── gcn_baseline/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── requirements.txt
│   ├── train.py
│   ├── gcn_best_model.pt
│   ├── gcn_config.json
│   └── reproduce_results.ipynb
├── .gitignore
└── README.md
```

Data is not committed. It is downloaded automatically by `torch_geometric` on first run and stored in a local `data/` directory (gitignored).

---

## Environment Setup

Each subdirectory has its own isolated environment. Set them up independently.

### Prerequisites
- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (for DimeNet++ and GCN baseline)

### DimeNet++
```bash
cd dimenet
uv sync          # creates dimenet/.venv — must be run once before use
source .venv/bin/activate
```

### GCN Baseline
```bash
cd gcn_baseline
uv sync
source .venv/bin/activate
```

### EGNN
```bash
cd egnn
pip install -r requirements.txt
```

### PaiNN
```bash
cd painn
pip install -r requirements.txt
```

Note: `torch-scatter`, `torch-sparse`, and `torch-cluster` require a `--find-links` flag matching your torch and CUDA versions. See `requirements.txt` for details.

---

## Reproducing Results

### DimeNet++ — evaluate saved checkpoint
```bash
cd dimenet
source .venv/bin/activate
jupyter notebook reproduce_results.ipynb
```
Loads `dimenet_best_model.pt` and evaluates on the test split. No retraining needed.

### DimeNet++ — retrain from scratch
Open `train.py` and run all cells in the training notebook.

### GCN Baseline — train
```bash
cd gcn_baseline
source .venv/bin/activate
python train.py                  # logs to Weights & Biases
python train.py --no_wandb       # disable W&B
```

Key arguments:
```
--data_root       path to QM9 data dir  (default: ../data/QM9)
--hidden_channels                       (default: 128)
--num_layers                            (default: 4)
--epochs                                (default: 300)
--patience        early stopping        (default: 30)
--checkpoint      output .pt file       (default: gcn_best_model.pt)
```

### EGNN — evaluate saved checkpoint
```bash
cd egnn
jupyter notebook reproduce_results.ipynb
```
Loads `egnn_best_model.pt` and evaluates on the test split. No retraining needed.

### EGNN — retrain from scratch
```bash
cd egnn
python train.py
```
Logs training metrics to Weights & Biases. Saves `egnn_best_model.pt` and `egnn_config.json` on completion.

### PaiNN — evaluate saved checkpoint
```bash
cd painn
jupyter notebook reproduce_results.ipynb
```
Loads `painn_best_model.pt` and evaluates on the test split. No retraining needed.

### PaiNN — retrain from scratch
```bash
cd painn
python train.py
```
Logs training metrics to Weights & Biases. Saves `painn_best_model.pt` and `painn_config.json` on completion.
