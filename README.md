# QM9 Molecular Property Prediction: GCN vs. DimeNet++

Codebase for the Course Mini-Project — **24-788 Introduction to Deep Learning (Spring 2026)**.

**Task**: Predict the HOMO-LUMO gap (Δε, target index 4) of small organic molecules from the QM9 dataset (~130k molecules), framed as a graph regression problem.

---

## Models

| Directory | Model | Description |
|---|---|---|
| `dimenet/` | DimeNet++ | Directional message-passing GNN using 3D bond lengths and angles |
| `gcn_baseline/` | GCN | Standard graph convolutional network; course baseline |

Both models are trained and evaluated on the same fixed split:
- **Train**: indices 0–109,999 (110k molecules)
- **Val**: indices 110,000–119,999 (10k molecules)
- **Test**: indices 120,000+ (~10.8k molecules)

Target normalization is computed from the training set only.

---

## Repository Structure

```
repo/
├── dimenet/
│   ├── pyproject.toml        # uv environment
│   ├── uv.lock
│   ├── requirements.txt      # human-readable deps
│   ├── dimenet2.ipynb        # main training notebook
│   ├── dimenet_best_model.pt # saved checkpoint
│   ├── dimenet_config.json   # hyperparameters + normalization stats
│   └── reproduce_results.ipynb
├── gcn_baseline/
│   ├── pyproject.toml        # uv environment (separate from dimenet)
│   ├── uv.lock
│   ├── requirements.txt
│   └── train.py              # training script
├── .gitignore
└── README.md
```

Data is not committed. It is downloaded automatically by `torch_geometric` on first run and stored in a local `data/` directory (gitignored).

---

## Environment Setup

Each subdirectory has its own isolated environment. Set them up independently.

### Prerequisites
- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### DimeNet++
```bash
cd dimenet
uv sync
source .venv/bin/activate
```

### GCN Baseline
```bash
cd gcn_baseline
uv sync
source .venv/bin/activate
```

Alternatively, install with `pip` using the pinned versions in `requirements.txt` (see the file for the `--find-links` flag needed for `torch-scatter` / `torch-sparse`).

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
Open `dimenet2.ipynb` and run all cells.

### GCN Baseline — train
```bash
cd gcn_baseline
source .venv/bin/activate
python train.py                  # logs to Weights & Biases
python train.py --no_wandb       # disable W&B
```

Key arguments:
```
--data_root     path to QM9 data dir  (default: ../data/QM9)
--hidden_channels               (default: 128)
--num_layers                    (default: 4)
--epochs                        (default: 300)
--patience      early stopping  (default: 30)
--checkpoint    output .pt file (default: gcn_best_model.pt)
```

Saves `gcn_best_model.pt` and `gcn_config.json` on completion.
