"""
Kaggle Notebook Training Script for CausalMamba Denoisers

Usage in Kaggle:
1. Upload FinancialDenoising folder to Kaggle Dataset
2. Add TRMwithQuant data as dataset
3. Create new GPU notebook (T4 or P100)
4. Run this script

Estimated time: 6-9 hours (T4 GPU)
"""

import os
import sys
import subprocess
from pathlib import Path
import time

# Kaggle paths (adjust based on your dataset setup)
PROJECT_ROOT = "/kaggle/working/FinancialDenoising"
DATA_PATH = "/kaggle/input/your-data/train_only.csv"  # Adjust this!

# Training configuration
CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "device": "cuda",
    "window_size": 60,
    "d_model": 128,
    "n_layers": 4,
    "guidance_weight": 0.1,
}

def train_cluster(cluster_id: int):
    """Train a single cluster."""
    print(f"\n{'='*80}")
    print(f"Training Cluster {cluster_id}")
    print(f"{'='*80}\n")

    start_time = time.time()

    cmd = [
        "python",
        f"{PROJECT_ROOT}/training/train_denoiser.py",
        "--cluster_id", str(cluster_id),
        "--data_path", DATA_PATH,
        "--epochs", str(CONFIG["epochs"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--device", CONFIG["device"],
        "--window_size", str(CONFIG["window_size"]),
        "--d_model", str(CONFIG["d_model"]),
        "--n_layers", str(CONFIG["n_layers"]),
        "--guidance_weight", str(CONFIG["guidance_weight"]),
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
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  CausalMamba Denoiser Training - All Clusters                ║
    ║  Architecture: Causal (no future leakage)                    ║
    ║  Normalization: Instance (scale-invariant)                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Check environment
    print("[INFO] Checking environment...")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Device: {CONFIG['device']}")

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Data file not found: {DATA_PATH}")
        print("Please update DATA_PATH in this script!")
        return

    # Change to project directory
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    # Train all clusters
    total_start = time.time()
    results = {}

    for cluster_id in range(7):
        success = train_cluster(cluster_id)
        results[cluster_id] = success

        if not success:
            print(f"\n[WARNING] Cluster {cluster_id} failed, continuing to next...")

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"\nResults:")
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
