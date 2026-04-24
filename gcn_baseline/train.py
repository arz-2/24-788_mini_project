"""
GCN baseline for QM9 molecular property prediction.
Target: index 4 (HOMO-LUMO gap, delta_epsilon, in eV).
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import wandb


TARGET_IDX = 4  # HOMO-LUMO gap (eV)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


def load_data(data_root, target_idx):
    dataset = QM9(root=data_root)

    # Same fixed split as DimeNet++ notebook
    train_set = dataset[:110000]
    val_set = dataset[110000:120000]
    test_set = dataset[120000:]

    # Normalize using training set statistics only
    train_targets = torch.stack([d.y[0, target_idx] for d in train_set])
    mean = train_targets.mean().item()
    std = train_targets.std().item()

    return train_set, val_set, test_set, mean, std


def train_epoch(model, loader, optimizer, device, mean, std, target_idx):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        x = batch.x.float()
        y = (batch.y[:, target_idx] - mean) / std
        pred = model(x, batch.edge_index, batch.batch)
        loss = F.l1_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, mean, std, target_idx):
    model.eval()
    total_mae = 0.0
    for batch in loader:
        batch = batch.to(device)
        x = batch.x.float()
        y = batch.y[:, target_idx]
        pred = model(x, batch.edge_index, batch.batch) * std + mean
        total_mae += (pred - y).abs().sum().item()
    return 1000 * total_mae / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/QM9")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default="gcn_best_model.pt")
    parser.add_argument("--config_out", type=str, default="gcn_config.json")
    parser.add_argument("--wandb_project", type=str, default="qm9-gcn-baseline")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading QM9 dataset...")
    train_set, val_set, test_set, mean, std = load_data(args.data_root, TARGET_IDX)
    print(f"  Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")
    print(f"  Target mean={mean:.4f}  std={std:.4f}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    in_channels = train_set[0].x.shape[1]
    model = GCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-5
    )

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, mean, std, TARGET_IDX)
        val_mae = evaluate(model, val_loader, device, mean, std, TARGET_IDX)
        scheduler.step(val_mae)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_MAE={val_mae:.2f} meV"
            f" | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        if not args.no_wandb:
            wandb.log({"train_loss": train_loss, "val_mae_meV": val_mae, "epoch": epoch})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), args.checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final test evaluation
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_mae = evaluate(model, test_loader, device, mean, std, TARGET_IDX)
    print(f"\nBest val MAE: {best_val_mae:.2f} meV")
    print(f"Test MAE:     {test_mae:.2f} meV")

    if not args.no_wandb:
        wandb.summary["best_val_mae_meV"] = best_val_mae
        wandb.summary["test_mae_meV"] = test_mae
        wandb.finish()

    config = {
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "target_idx": TARGET_IDX,
        "target_mean": mean,
        "target_std": std,
        "best_val_mae": best_val_mae / 1000,
        "test_mae": test_mae / 1000,
    }
    with open(args.config_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {args.config_out}")


if __name__ == "__main__":
    main()
