"""
Random Forest anomaly detector for frequency monitoring.

This script trains a Random Forest classifier to detect anomalies
in frequency data from DER nodes.

Usage:
    # Train model on prepared data
    python -m data.random_forest --train data/rf_training_data.csv --save models/rf_model.pkl
    
    # Evaluate model
    python -m data.random_forest --train data/rf_training_data.csv --evaluate
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Feature columns used for training
FEATURE_COLS = [
    'freq_deviation',
    'freq_rolling_mean', 
    'freq_rolling_std',
    'freq_rate_change',
    'freq_abs_rate_change'
]

def parse_args():
    ap = argparse.ArgumentParser(description="Train Random Forest anomaly detector")
    ap.add_argument("--train", required=True, help="Path to training data CSV")
    ap.add_argument("--save", default="models/rf_model.pkl", help="Path to save trained model")
    ap.add_argument("--evaluate", action="store_true", help="Print evaluation metrics")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    ap.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    return ap.parse_args()

def load_training_data(csv_path: Path):
    """Load and validate training data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    missing = [col for col in FEATURE_COLS + ['label'] if col not in df.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")
    
    return df

def train_model(df: pd.DataFrame, n_estimators: int, test_size: float, random_state: int):
    """Train Random Forest classifier."""
    
    # Prepare features (X) and labels (y)
    X = df[FEATURE_COLS].values
    y = df['label'].values
    
    print(f"[train] Training data shape: {X.shape}")
    print(f"[train] Normal samples: {(y == 0).sum()}")
    print(f"[train] Anomaly samples: {(y == 1).sum()}")
    
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
    """Print evaluation metrics."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print("KEY METRICS FOR ANOMALY DETECTION")
    print("="*60)
    print(f"Precision (Anomaly): {precision:.4f} - When we predict anomaly, we're right {precision*100:.1f}% of time")
    print(f"Recall (Anomaly):    {recall:.4f} - We catch {recall*100:.1f}% of all anomalies")
    print(f"F1-Score (Anomaly):  {f1:.4f} - Overall anomaly detection performance")
    print(f"False Positives:     {fp} - Normal data incorrectly flagged as anomaly")
    print(f"False Negatives:     {fn} - Anomalies we missed")
    print("="*60 + "\n")

def save_model(model, save_path: Path):
    """Save trained model to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[train] Model saved to {save_path}")

def main():
    args = parse_args()
    
    print(f"[train] Loading training data from {args.train}...")
    df = load_training_data(Path(args.train))
    
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
    
    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_model(y_test, y_pred)
    
    # Save model
    save_model(model, Path(args.save))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {args.save}")
    print(f"\nTo use this model for real-time detection:")
    print(f"  1. Load model: model = pickle.load(open('{args.save}', 'rb'))")
    print(f"  2. Prepare features from new frequency data")
    print(f"  3. Predict: model.predict(features)")
    print("="*60)

if __name__ == "__main__":
    main()
