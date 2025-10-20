"""
Random Forest anomaly detector for frequency monitoring.

This script trains a Random Forest classifier to detect anomalies
in frequency data using moving average deviations.

Usage:
    # Train model on mod.csv (pre-labeled with anomalies)
    python -m data.random_forest --train data/mod.csv --save models/rf_model.pkl
    
    # Evaluate model
    python -m data.random_forest --train data/mod.csv --evaluate
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Feature columns - deviations from moving averages
FEATURE_COLS = [
    'deviation_from_ma60',
    'deviation_from_ma120',
    'deviation_from_ma180',
    'deviation_from_ma240'
]

def parse_args():
    ap = argparse.ArgumentParser(description="Train Random Forest anomaly detector")
    ap.add_argument("--train", required=True, help="Path to training data CSV (mod.csv)")
    ap.add_argument("--save", default="models/rf_model.pkl", help="Path to save trained model")
    ap.add_argument("--evaluate", action="store_true", help="Print evaluation metrics")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    ap.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    return ap.parse_args()

def load_and_prepare_data(csv_path: Path):
    """Load mod.csv and engineer deviation features."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")
    
    print(f"[train] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required = ['Hz_adj', 'Hz_modified', 'ma60', 'ma120', 'ma180', 'ma240']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    print(f"[train] Loaded {len(df)} rows")
    
    # Convert Hz_modified (TRUE/FALSE) to labels (1/0)
    df['label'] = df['Hz_modified'].map({'TRUE': 1, True: 1, 'FALSE': 0, False: 0}).astype(int)
    
    # Engineer features: deviation from each moving average
    df['deviation_from_ma60'] = df['Hz_adj'] - df['ma60']
    df['deviation_from_ma120'] = df['Hz_adj'] - df['ma120']
    df['deviation_from_ma180'] = df['Hz_adj'] - df['ma180']
    df['deviation_from_ma240'] = df['Hz_adj'] - df['ma240']
    
    # Drop any NaN rows
    df = df.dropna(subset=FEATURE_COLS + ['label'])
    
    print(f"[train] After cleaning: {len(df)} rows")
    print(f"[train] Frequency range: {df['Hz_adj'].min():.4f} - {df['Hz_adj'].max():.4f} Hz")
    
    return df

def train_model(df: pd.DataFrame, n_estimators: int, test_size: float, random_state: int):
    """Train Random Forest classifier."""
    
    # Prepare features (X) and labels (y)
    X = df[FEATURE_COLS].values
    y = df['label'].values
    
    print(f"[train] Training data shape: {X.shape}")
    print(f"[train] Normal samples: {(y == 0).sum()}")
    print(f"[train] Anomaly samples: {(y == 1).sum()}")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both normal and anomaly samples")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[train] Training set: {len(X_train)} samples")
    print(f"[train] Test set: {len(X_test)} samples")
    
    # Train Random Forest
    print(f"[train] Training Random Forest with {n_estimators} trees...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("[train] Training complete!")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    # Evaluation metrics
    print("MODEL EVALUATION")
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"Actual Anomaly  {cm[1,0]:6d}  {cm[1,1]:7d}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Anomaly'],
        digits=4
    ))
    
    # Calculate key metrics manually for emphasis
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("KEY METRICS FOR ANOMALY DETECTION")
    print(f"Precision (Anomaly): {precision:.4f} - When we predict anomaly, we're right {precision*100:.1f}% of time")
    print(f"Recall (Anomaly):    {recall:.4f} - We catch {recall*100:.1f}% of all anomalies")
    print(f"F1-Score (Anomaly):  {f1:.4f} - Overall anomaly detection performance")
    print(f"False Positives:     {fp} - Normal data incorrectly flagged as anomaly")
    print(f"False Negatives:     {fn} - Anomalies we missed")

def save_model(model, save_path: Path):
    """Save trained model to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[train] Model saved to {save_path}")

def main():
    args = parse_args()
    
    # Load and prepare data
    df = load_and_prepare_data(Path(args.train))
    
    # Train model
    model, X_test, y_test, y_pred = train_model(
        df, 
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Deviation from Moving Averages):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Evaluate
    if args.evaluate:
        evaluate_model(y_test, y_pred)
    
    # Save model
    save_model(model, Path(args.save))
    
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {args.save}")
    print(f"\nModel uses deviations from moving averages (ma60, ma120, ma180, ma240)")
    print(f"to detect when frequency varies abnormally from expected values.")

if __name__ == "__main__":
    main()
