"""
DimeNet++ training on QM9 — predict HOMO-LUMO gap (target index 4).
"""

import json
import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DimeNetPlusPlus
from tqdm import tqdm
import wandb

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_IDX = 4  # HOMO-LUMO gap (eV)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model_config = {
    "hidden_channels":  64,
    "out_channels":     1,
    "num_blocks":       4,
    "num_spherical":    4,
    "num_radial":       4,
    "cutoff":           5.0,
    "int_emb_size":     64,
    "out_emb_channels": 64,
    "basis_emb_size":   32,
}

batch_size  = 128
num_epochs  = 50
lr          = 1e-3
num_workers = 4
checkpoint  = "dimenet_best_model.pt"
config_file = "dimenet_config.json"
log_file    = "../training.log"
wandb_project = "dimenet-qm9"

# ── Dataset ────────────────────────────────────────────────────────────────────
print(f"Using device: {device} ")

dataset = QM9(root="../data/QM9")

train_dataset = dataset[:110000]
val_dataset   = dataset[110000:120000]
test_dataset  = dataset[120000:]

print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

train_targets = torch.stack([d.y[0, TARGET_IDX] for d in train_dataset])
target_mean   = train_targets.mean().item()
target_std    = train_targets.std().item()

def normalize(y):
    return (y - target_mean) / target_std

def denormalize(y_norm):
    return y_norm * target_std + target_mean

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ── Model ──────────────────────────────────────────────────────────────────────
model = DimeNetPlusPlus(**model_config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# ── Train / Eval functions ─────────────────────────────────────────────────────
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        y   = normalize(batch.y[:, TARGET_IDX])
        out = model(batch.z, batch.pos, batch.batch)
        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_mae = 0.0
    for batch in loader:
        batch = batch.to(device)
        out    = model(batch.z, batch.pos, batch.batch)
        preds  = denormalize(out.squeeze())
        labels = batch.y[:, TARGET_IDX]
        total_mae += (preds - labels).abs().sum().item()
    return 1000 * total_mae / len(loader.dataset)  # eV -> meV

# ── wandb ──────────────────────────────────────────────────────────────────────
wandb.init(project=wandb_project, config={**model_config, "batch_size": batch_size,
                                           "epochs": num_epochs, "lr": lr})

# ── Training loop ──────────────────────────────────────────────────────────────
best_val_mae = float("inf")

log = open(log_file, "w", buffering=1)

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader)
    val_mae    = evaluate(model, val_loader)
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    line = f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} meV | LR: {current_lr:.2e}"
    print(line)
    log.write(line + "\n")

    wandb.log({"train_loss": train_loss, "val_mae_meV": val_mae, "lr": current_lr, "epoch": epoch})

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), checkpoint)

# ── Final evaluation ───────────────────────────────────────────────────────────
model.load_state_dict(torch.load(checkpoint, map_location=device))
test_mae = evaluate(model, test_loader)

final = f"\nFinal Test MAE: {test_mae:.2f} meV"
print(final)
log.write(final + "\n")
log.close()

wandb.summary["best_val_mae_meV"] = best_val_mae
wandb.summary["test_mae_meV"]     = test_mae
wandb.finish()

# ── Save config ────────────────────────────────────────────────────────────────
with open(config_file, "w") as f:
    json.dump({**model_config, "target_idx": TARGET_IDX,
               "target_mean": target_mean, "target_std": target_std,
               "best_val_mae_meV": best_val_mae, "test_mae_meV": test_mae}, f, indent=2)
