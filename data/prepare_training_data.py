"""
Prepare labeled training data for Random Forest anomaly detection.

Reads normalized.csv, engineers features from frequency, and injects
synthetic anomalies to create a labeled dataset for training.

Usage:
    python -m data.prepare_training_data --csv data/normalized.csv --output data/rf_training_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare RF training data with synthetic anomalies")
    ap.add_argument("--csv", default="data/normalized.csv", help="Path to normalized CSV")
    ap.add_argument("--output", default="data/rf_training_data.csv", help="Output path for training data")
    ap.add_argument("--anomaly-ratio", type=float, default=0.15, help="Fraction of data to label as anomalies")
    return ap.parse_args()

def engineer_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Engineer features from frequency column for anomaly detection.
    
    Features created:
    - freq_deviation: How far from 60 Hz
    - freq_rolling_mean: Moving average
    - freq_rolling_std: Moving standard deviation
    - freq_rate_change: How fast frequency is changing
    - freq_abs_rate_change: Absolute rate of change
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Deviation from nominal 60 Hz
    df['freq_deviation'] = df['frequency_hz'] - 60.0
    
    # Rolling statistics
    df['freq_rolling_mean'] = df['frequency_hz'].rolling(window=window, min_periods=1).mean()
    df['freq_rolling_std'] = df['frequency_hz'].rolling(window=window, min_periods=1).std().fillna(0)
    
    # Rate of change (difference between consecutive readings)
    df['freq_rate_change'] = df['frequency_hz'].diff().fillna(0)
    df['freq_abs_rate_change'] = df['freq_rate_change'].abs()
    
    # Fill any remaining NaNs
    df = df.fillna(0)
    
    return df

def inject_anomalies(df: pd.DataFrame, anomaly_ratio: float = 0.15) -> pd.DataFrame:
    """
    Inject synthetic anomalies into the dataset.
    
    Anomaly types:
    1. Bias attack: Sudden offset from 60 Hz
    2. Drift attack: Gradual drift away from 60 Hz
    3. Spike attack: Random spikes
    4. Oscillation attack: Rapid oscillations
    """
    df = df.copy()
    n_rows = len(df)
    n_anomalies = int(n_rows * anomaly_ratio)
    
    # Initialize all as normal
    df['label'] = 0
    df['anomaly_type'] = 'normal'
    
    # Randomly select indices for anomalies
    anomaly_indices = np.random.choice(n_rows, size=n_anomalies, replace=False)
    
    # Split anomalies into different types
    n_per_type = n_anomalies // 4
    bias_idx = anomaly_indices[:n_per_type]
    drift_idx = anomaly_indices[n_per_type:2*n_per_type]
    spike_idx = anomaly_indices[2*n_per_type:3*n_per_type]
    osc_idx = anomaly_indices[3*n_per_type:]
    
    # 1. Bias attack: Add constant offset
    for idx in bias_idx:
        bias = np.random.choice([-0.3, -0.2, 0.2, 0.3])  # Significant deviation
        df.loc[idx, 'frequency_hz'] += bias
        df.loc[idx, 'label'] = 1
        df.loc[idx, 'anomaly_type'] = 'bias'
    
    # 2. Drift attack: Gradual drift over 10-30 samples
    for start_idx in drift_idx:
        drift_length = np.random.randint(10, 30)
        end_idx = min(start_idx + drift_length, n_rows)
        drift_amount = np.random.choice([-0.4, 0.4])
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            # Linear drift
            df.loc[idx, 'frequency_hz'] += (drift_amount * i / drift_length)
            df.loc[idx, 'label'] = 1
            df.loc[idx, 'anomaly_type'] = 'drift'
    
    # 3. Spike attack: Random large spikes
    for idx in spike_idx:
        spike = np.random.choice([-0.5, -0.4, 0.4, 0.5])
        df.loc[idx, 'frequency_hz'] += spike
        df.loc[idx, 'label'] = 1
        df.loc[idx, 'anomaly_type'] = 'spike'
    
    # 4. Oscillation attack: Rapid back-and-forth
    for start_idx in osc_idx:
        osc_length = np.random.randint(5, 15)
        end_idx = min(start_idx + osc_length, n_rows)
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            # Alternating +/-
            osc = 0.2 * (1 if i % 2 == 0 else -1)
            df.loc[idx, 'frequency_hz'] += osc
            df.loc[idx, 'label'] = 1
            df.loc[idx, 'anomaly_type'] = 'oscillation'
    
    return df

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"[prepare_training] Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"[prepare_training] Loaded {len(df)} rows")
    print(f"[prepare_training] Original frequency range: {df['frequency_hz'].min():.4f} - {df['frequency_hz'].max():.4f} Hz")
    
    # Inject synthetic anomalies first
    print(f"[prepare_training] Injecting synthetic anomalies ({args.anomaly_ratio*100:.1f}% of data)...")
    df = inject_anomalies(df, anomaly_ratio=args.anomaly_ratio)
    
    n_anomalies = (df['label'] == 1).sum()
    print(f"[prepare_training] Injected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)")
    
    # Engineer features
    print(f"[prepare_training] Engineering features...")
    df = engineer_features(df)
    
    print(f"[prepare_training] New frequency range (with anomalies): {df['frequency_hz'].min():.4f} - {df['frequency_hz'].max():.4f} Hz")
    
    # Save to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[prepare_training] Saved training data to {output_path}")
    print(f"\nFeatures created:")
    print(f"  - freq_deviation")
    print(f"  - freq_rolling_mean")
    print(f"  - freq_rolling_std")
    print(f"  - freq_rate_change")
    print(f"  - freq_abs_rate_change")
    print(f"\nAnomaly distribution:")
    print(df['anomaly_type'].value_counts())
    print(f"\nReady to train! Run:")
    print(f"  python -m data.random_forest --train data/rf_training_data.csv")

if __name__ == "__main__":
    main()
