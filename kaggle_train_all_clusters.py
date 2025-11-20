"""
Kaggle Notebook Training Script for CausalMamba Denoisers

Usage in Kaggle:
1. Upload FinancialDenoising folder (with train.csv) to Kaggle Dataset
2. Create new GPU notebook (T4 or P100)
3. Run this script

Estimated time: 6-9 hours (T4 GPU)

This script will:
- Load train.csv
- Split into train/val (80/20, Purged Embargo Walk-Forward)
- Train 7 CausalMamba models (one per feature cluster)
- Save trained models to trained_models/
"""

import os
import sys
import subprocess
from pathlib import Path
import time
import argparse

# Kaggle paths (auto-detected from script location)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "train.csv")

# Training configuration
CONFIG = {
    "epochs": 100,
    "lr": 5e-4,          # Learning rate
    "batch_size": 32,
    "device": "cuda",
    "window_size": 60,
    "d_model": 128,
    "n_layers": 4,
    "guidance_weight": 0.1,
    "target_loss": 0.2,  # Early stopping: stop when loss reaches this value
    "patience": 15,      # Early stopping: stop if no improvement for N epochs
    "train_ratio": 0.8,  # 80% train, 20% val
    "embargo_days": 0,   # Buffer days between train/val
}

def train_cluster(cluster_id: int, train_data_path: str):
    """Train a single cluster."""
    print(f"\n{'='*80}")
    print(f"Training Cluster {cluster_id}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Auto-detect paths relative to PROJECT_ROOT
    cluster_config_path = os.path.join(PROJECT_ROOT, "clustering_results", "cluster_assignments.json")
    output_dir = os.path.join(PROJECT_ROOT, "trained_models")

    cmd = [
        "python",
        os.path.join(PROJECT_ROOT, "training", "train_denoiser.py"),
        "--cluster_id", str(cluster_id),
        "--data_path", train_data_path,
        "--cluster_config", cluster_config_path,
        "--output_dir", output_dir,
        "--epochs", str(CONFIG["epochs"]),
        "--lr", str(CONFIG["lr"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--device", CONFIG["device"],
        "--window_size", str(CONFIG["window_size"]),
        "--d_model", str(CONFIG["d_model"]),
        "--n_layers", str(CONFIG["n_layers"]),
        "--guidance_weight", str(CONFIG["guidance_weight"]),
        "--target_loss", str(CONFIG["target_loss"]),
        "--patience", str(CONFIG["patience"]),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n[OK] Cluster {cluster_id} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Cluster {cluster_id} failed: {e}")
        return False

def main():
    """Train all 7 clusters sequentially."""
    parser = argparse.ArgumentParser(description="Train all CausalMamba denoiser models")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="Device (cuda/cpu)")
    parser.add_argument("--window_size", type=int, default=CONFIG["window_size"], help="Window size")
    parser.add_argument("--d_model", type=int, default=CONFIG["d_model"], help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=CONFIG["n_layers"], help="Number of layers")
    parser.add_argument("--guidance_weight", type=float, default=CONFIG["guidance_weight"], help="Guidance loss weight")
    parser.add_argument("--target_loss", type=float, default=CONFIG["target_loss"], help="Early stopping target loss")
    parser.add_argument("--patience", type=int, default=CONFIG["patience"], help="Early stopping patience")

    # Data split parameters
    parser.add_argument("--train_ratio", type=float, default=CONFIG["train_ratio"], help="Train/val split ratio")
    parser.add_argument("--embargo_days", type=int, default=CONFIG["embargo_days"], help="Embargo days between train/val")

    args = parser.parse_args()

    # Override CONFIG with command-line arguments
    CONFIG["epochs"] = args.epochs
    CONFIG["lr"] = args.lr
    CONFIG["batch_size"] = args.batch_size
    CONFIG["device"] = args.device
    CONFIG["window_size"] = args.window_size
    CONFIG["d_model"] = args.d_model
    CONFIG["n_layers"] = args.n_layers
    CONFIG["guidance_weight"] = args.guidance_weight
    CONFIG["target_loss"] = args.target_loss
    CONFIG["patience"] = args.patience
    CONFIG["train_ratio"] = args.train_ratio
    CONFIG["embargo_days"] = args.embargo_days

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  CausalMamba Denoiser Training - All Clusters                ║
    ║  Architecture: Causal (no future leakage)                    ║
    ║  Normalization: Instance (scale-invariant)                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Change to project directory
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    # Check environment
    print("[INFO] Checking environment...")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Train CSV: {TRAIN_CSV_PATH}")
    print(f"  Device: {CONFIG['device']}")

    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"\n[ERROR] Data file not found: {TRAIN_CSV_PATH}")
        print("Please ensure train.csv is in the uploaded dataset!")
        return

    # Step 1: Load and split data
    print(f"\n{'='*80}")
    print("Step 1: Loading and splitting data")
    print(f"{'='*80}\n")

    try:
        # Import data utilities
        from utils.data_utils import load_and_split

        # Load and split
        train_df, val_df = load_and_split(
            TRAIN_CSV_PATH,
            train_ratio=CONFIG["train_ratio"],
            embargo_days=CONFIG["embargo_days"],
            save_splits=True,
            output_dir=PROJECT_ROOT
        )

        train_data_path = f"{PROJECT_ROOT}/train_only.csv"
        val_data_path = f"{PROJECT_ROOT}/val_only.csv"

        print(f"\n[OK] Data split completed")
        print(f"  Train: {train_data_path}")
        print(f"  Val: {val_data_path}")

    except Exception as e:
        print(f"\n[ERROR] Failed to split data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Train all clusters
    print(f"\n{'='*80}")
    print("Step 2: Training all clusters")
    print(f"{'='*80}\n")

    total_start = time.time()
    results = {}

    for cluster_id in range(7):
        success = train_cluster(cluster_id, train_data_path)
        results[cluster_id] = success

        if not success:
            print(f"\n[WARNING] Cluster {cluster_id} failed, continuing to next...")

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"\nData Split:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"\nTraining Results:")
    for cluster_id, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  Cluster {cluster_id}: {status}")

    successful = sum(results.values())
    print(f"\nSuccessful: {successful}/7 clusters")

    # Save artifacts info
    print(f"\n[INFO] Model checkpoints saved to:")
    print(f"  {PROJECT_ROOT}/trained_models/")
    print("\nTo download in Kaggle:")
    print("  1. Click 'Save Version' → 'Save & Run All'")
    print("  2. After completion, download output folder")
    print("  3. Extract trained_models/ directory")

if __name__ == "__main__":
    main()
