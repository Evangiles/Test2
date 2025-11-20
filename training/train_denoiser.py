"""
Train Group-Specific Mamba Denoisers

Trains separate denoiser models for each feature cluster
with cluster-specific guidance strategies.

Usage:
    python train/train_group_denoiser.py --cluster_id 0 --epochs 100
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_mamba import (
    CausalMambaDenoiser,
    VPSDE,
    DenoisingLoss
)


class TimeSeriesWindowDataset(Dataset):
    """
    Dataset for windowed multivariate time series with INSTANCE NORMALIZATION.

    Creates sliding windows from time series data and normalizes each window
    independently using its own statistics. This ensures:
    1. No data leakage between train/val
    2. Scale invariance (robust to price regime changes)
    3. Model learns relative patterns, not absolute values

    Key difference from global normalization:
    - Global: window_norm = (window - train_mean) / train_std  (leaks scale info)
    - Instance: window_norm = (window - window.mean()) / window.std()  (scale-free)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list,
        window_size: int = 60,
        stride: int = 1
    ):
        """
        Args:
            data: DataFrame with time series
            feature_cols: List of feature column names
            window_size: Window size
            stride: Stride for sliding window
        """
        self.window_size = window_size
        self.feature_cols = feature_cols

        # Extract feature data (keep raw, normalize in __getitem__)
        feature_data = data[feature_cols].values  # [T, F]

        # Handle NaN and inf
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Create windows (store raw data, normalize per-instance)
        self.windows = []
        for i in range(0, len(feature_data) - window_size + 1, stride):
            window = feature_data[i:i + window_size]
            self.windows.append(window)

        self.windows = np.array(self.windows, dtype=np.float32)  # [N, W, F]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Returns instance-normalized window.

        Normalization is applied PER WINDOW to ensure:
        - No leakage of global statistics
        - Robustness to price regime changes
        - Model learns shape, not absolute scale
        """
        window = self.windows[idx].copy()  # [W, F]

        # Instance normalization: use window's own statistics
        window_mean = window.mean()
        window_std = window.std()

        # Normalize
        window_norm = (window - window_mean) / (window_std + 1e-6)

        return torch.FloatTensor(window_norm)  # [W, F]


def load_cluster_config(config_path: Path):
    """Load cluster assignments from JSON."""
    with open(config_path, 'r') as f:
        cluster_config = json.load(f)
    return cluster_config


def get_cluster_features(cluster_config: dict, cluster_id: int):
    """Extract feature names for a specific cluster."""
    cluster_key = f"cluster_{cluster_id}"
    if cluster_key not in cluster_config['clusters']:
        raise ValueError(f"Cluster {cluster_id} not found")

    cluster_info = cluster_config['clusters'][cluster_key]
    return cluster_info['features'], cluster_info.get('type', 'random_walk')


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    sde: VPSDE,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    accumulation_steps: int = 1
):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss_sum = 0.0
    mse_loss_sum = 0.0
    guide_loss_sum = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        x0 = batch.to(device)  # [B, W, F]

        # Sample random timesteps
        t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)

        # Forward diffusion: add noise
        x_t, noise = sde.sample(x0, t)

        # Predict noise
        predicted_noise = model(x_t, t)

        # Compute predicted x0 for guidance loss
        alpha_bar = sde.alphas_cumprod[t]
        while alpha_bar.dim() < x_t.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        predicted_x0 = (x_t - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / (torch.sqrt(alpha_bar) + 1e-8)
        predicted_x0 = torch.clamp(predicted_x0, -10, 10)  # Prevent extreme values

        # Compute loss (returns 3 values)
        total_loss, mse_loss, guide_loss = loss_fn(predicted_noise, noise, predicted_x0)

        # Check for NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n⚠️  NaN/Inf detected in loss! Skipping batch...")
            continue

        # Scale loss for gradient accumulation
        total_loss_scaled = total_loss / accumulation_steps

        # Backward
        total_loss_scaled.backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Strong gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate losses (unscaled)
        total_loss_sum += total_loss.item()
        mse_loss_sum += mse_loss.item()
        guide_loss_sum += guide_loss.item()

    n_batches = len(dataloader)
    return (
        mse_loss_sum / n_batches,
        guide_loss_sum / n_batches,
        total_loss_sum / n_batches
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", type=int, default=None, help="Cluster ID to train (0-6). If not specified, trains all clusters.")
    parser.add_argument("--data_path", type=str, default="train_only.csv", help="Path to training data CSV")
    parser.add_argument("--cluster_config", type=str, default="clustering_results/cluster_assignments.json", help="Path to cluster config JSON")
    parser.add_argument("--output_dir", type=str, default="trained_models", help="Directory to save trained models")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--guidance_weight", type=float, default=0.1)
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--target_loss", type=float, default=None, help="Stop when loss reaches this value (early stopping)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum loss improvement to reset patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("="*80)
    print("GROUP-SPECIFIC DENOISER TRAINING")
    print("="*80)

    # Load cluster configuration
    cluster_config = load_cluster_config(Path(args.cluster_config))

    # If cluster_id not specified, train all clusters
    if args.cluster_id is None:
        n_clusters = cluster_config['metadata']['n_clusters']
        print(f"\nNo cluster_id specified → Training all {n_clusters} clusters")
        print(f"Device: {args.device}")

        import subprocess
        import sys

        for cluster_id in range(n_clusters):
            print(f"\n{'='*80}")
            print(f"Training Cluster {cluster_id}/{n_clusters-1}")
            print(f"{'='*80}")

            cmd = [
                sys.executable,
                __file__,
                "--cluster_id", str(cluster_id),
                "--data_path", args.data_path,
                "--cluster_config", args.cluster_config,
                "--output_dir", args.output_dir,
                "--window_size", str(args.window_size),
                "--d_model", str(args.d_model),
                "--n_layers", str(args.n_layers),
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--guidance_weight", str(args.guidance_weight),
                "--accumulation_steps", str(args.accumulation_steps),
                "--patience", str(args.patience),
                "--min_delta", str(args.min_delta),
                "--device", args.device,
            ]

            if args.target_loss is not None:
                cmd.extend(["--target_loss", str(args.target_loss)])

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n[ERROR] Cluster {cluster_id} failed!")

        print(f"\n{'='*80}")
        print(f"All clusters training completed!")
        print(f"{'='*80}")
        return

    print(f"\nCluster ID: {args.cluster_id}")
    print(f"Device: {args.device}")

    # Get cluster features
    feature_names, cluster_type = get_cluster_features(cluster_config, args.cluster_id)

    print(f"Cluster type: {cluster_type}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {', '.join(feature_names[:10])}")
    if len(feature_names) > 10:
        print(f"          ... and {len(feature_names)-10} more")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows")

    # Validate that we're using train-only data
    if 'train_only' not in str(args.data_path):
        print(f"\n⚠️  WARNING: Training on {args.data_path}")
        print(f"    Expected 'train_only.csv' to prevent validation leakage!")
        print(f"    Proceeding anyway, but ensure this is intentional.\n")

    # Create dataset with INSTANCE NORMALIZATION (no global statistics needed!)
    print(f"\nCreating dataset...")
    print(f"  Using Instance Normalization (per-window statistics)")
    print(f"  This ensures scale-invariance and prevents data leakage")

    dataset = TimeSeriesWindowDataset(
        df,
        feature_names,
        window_size=args.window_size,
        stride=1
    )
    print(f"Created {len(dataset)} windows")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False
    )

    # Initialize CAUSAL model (trading-ready, no future leakage)
    print(f"\nInitializing CausalMambaDenoiser...")
    print(f"  This model uses forward-only Mamba (no backward pass)")
    print(f"  Suitable for real-time trading deployment")

    model = CausalMambaDenoiser(
        n_features=len(feature_names),
        window_size=args.window_size,
        d_model=args.d_model,
        n_layers=args.n_layers
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Initialize SDE
    sde = VPSDE(device=args.device)

    # Initialize loss
    loss_fn = DenoisingLoss(
        cluster_type=cluster_type,
        guidance_weight=args.guidance_weight
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler with warmup
    def get_lr_scale(epoch, warmup_epochs=5):
        """Linear warmup then constant"""
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    # Training loop with early stopping
    print(f"\nStarting training (max {args.epochs} epochs)...")
    if args.target_loss is not None:
        print(f"  Target MSE loss: {args.target_loss:.6f} (early stop when reached)")
    print(f"  Patience: {args.patience} epochs (stop if no improvement)")

    best_mse = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Apply learning rate warmup
        lr_scale = get_lr_scale(epoch, warmup_epochs=5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * lr_scale

        avg_mse, avg_guide, avg_total = train_epoch(model, dataloader, sde, loss_fn, optimizer, args.device, args.accumulation_steps)

        print(f"Epoch {epoch+1}/{args.epochs} - MSE: {avg_mse:.6f}, Guidance: {avg_guide:.6f}, Total: {avg_total:.6f}, LR: {args.lr * lr_scale:.6f}")

        # Check for improvement (based on MSE only)
        if avg_mse < best_mse - args.min_delta:
            best_mse = avg_mse
            epochs_without_improvement = 0

            # Save best model
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / f"cluster_{args.cluster_id}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mse_loss': avg_mse,
                'guidance_loss': avg_guide,
                'total_loss': avg_total,
                'cluster_id': args.cluster_id,
                'cluster_type': cluster_type,
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'window_size': args.window_size,
                'd_model': args.d_model,
                'n_layers': args.n_layers,
                'normalization_type': 'instance',
            }, checkpoint_path)

            print(f"  → Saved checkpoint to {checkpoint_path}")

            # Check if target MSE loss reached
            if args.target_loss is not None and avg_mse <= args.target_loss:
                print(f"\n[Early Stop] Target MSE loss {args.target_loss:.6f} reached!")
                print(f"  Final MSE: {avg_mse:.6f} at epoch {epoch+1}")
                break
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

            # Check patience
            if epochs_without_improvement >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs")
                print(f"  Best MSE: {best_mse:.6f}")
                break

    print(f"\nTraining completed! Best MSE: {best_mse:.6f}")


if __name__ == "__main__":
    main()
