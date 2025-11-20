"""
Data Utilities for Financial Denoising

Includes train/val split with Purged Embargo Walk-Forward methodology
to prevent data leakage in time series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def purged_embargo_split(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    embargo_days: int = 0,
    date_column: str = 'date_id'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purged Embargo Walk-Forward split for time series.

    Ensures no data leakage by:
    1. Chronological split (train → val, no shuffle)
    2. Optional embargo period (buffer zone between train/val)

    Args:
        data: Input dataframe with time series
        train_ratio: Fraction of data for training (default 0.8)
        embargo_days: Number of days to skip between train/val (default 0)
        date_column: Name of date column for sorting (default 'date_id')

    Returns:
        train_df, val_df: Split dataframes

    Example:
        >>> df = pd.read_csv('train.csv')
        >>> train, val = purged_embargo_split(df, train_ratio=0.8, embargo_days=5)
        >>> print(f"Train: {len(train)}, Val: {len(val)}")
    """
    # Ensure chronological order
    data = data.sort_values(date_column).reset_index(drop=True)

    n_total = len(data)
    n_train = int(n_total * train_ratio)

    # Split indices
    train_end = n_train
    val_start = train_end + embargo_days

    # Create splits
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[val_start:].copy()

    print(f"Purged Embargo Walk-Forward Split:")
    print(f"  Total rows: {n_total}")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/n_total*100:.1f}%)")
    if embargo_days > 0:
        print(f"  Embargo: {embargo_days} rows (buffer zone)")
    print(f"  Validation: {len(val_df)} rows ({len(val_df)/n_total*100:.1f}%)")

    # Verify no overlap
    if len(train_df) > 0 and len(val_df) > 0:
        last_train_date = train_df[date_column].iloc[-1]
        first_val_date = val_df[date_column].iloc[0]
        print(f"  Last train date: {last_train_date}")
        print(f"  First val date: {first_val_date}")

        if first_val_date <= last_train_date:
            print(f"  WARNING: Temporal overlap detected!")
        else:
            print(f"  [OK] No temporal overlap")

    return train_df, val_df


def save_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> Tuple[Path, Path]:
    """
    Save train/val split to CSV files.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        output_dir: Directory to save files
        prefix: Optional prefix for filenames

    Returns:
        train_path, val_path: Paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{prefix}train_only.csv"
    val_path = output_dir / f"{prefix}val_only.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nSaved splits:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    return train_path, val_path


def load_and_split(
    data_path: str,
    train_ratio: float = 0.8,
    embargo_days: int = 0,
    save_splits: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV and split into train/val with optional saving.

    Args:
        data_path: Path to input CSV (e.g., 'train.csv')
        train_ratio: Fraction for training (default 0.8)
        embargo_days: Buffer days between train/val (default 0)
        save_splits: Whether to save split files (default False)
        output_dir: Directory to save splits (required if save_splits=True)

    Returns:
        train_df, val_df: Split dataframes

    Example:
        >>> train, val = load_and_split('train.csv', train_ratio=0.8, embargo_days=5)
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} rows")

    # Perform split
    train_df, val_df = purged_embargo_split(
        data,
        train_ratio=train_ratio,
        embargo_days=embargo_days
    )

    # Optional: save splits
    if save_splits:
        if output_dir is None:
            output_dir = Path(data_path).parent
        save_split(train_df, val_df, output_dir)

    return train_df, val_df


if __name__ == "__main__":
    """Test split functionality."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--embargo_days", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".")

    args = parser.parse_args()

    train_df, val_df = load_and_split(
        args.data_path,
        train_ratio=args.train_ratio,
        embargo_days=args.embargo_days,
        save_splits=args.save,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("Split Summary")
    print("="*80)
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"\nTrain date range: {train_df['date_id'].min()} - {train_df['date_id'].max()}")
    print(f"Val date range: {val_df['date_id'].min()} - {val_df['date_id'].max()}")
