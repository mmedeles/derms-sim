import numpy as np
from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier

class Node_RF:
    """
    Random Forest anomaly detector placeholder for a single node.
    """
    def __init__(self):
        # Initialize Random Forest
        self.model = RandomForestClassifier(n_estimators=100)
        self.trained = False
        self.features: List[List[float]] = []
        self.labels: List[int] = []

    def add_training_sample(self, feature_vector: List[float], label: int):
        # Add sample to training set
        self.features.append(feature_vector)
        self.labels.append(label)

    def train_model(self):
        # Train RF model if enough samples
        if len(self.features) < 5:
            return
        X = np.array(self.features)
        y = np.array(self.labels)
        self.model.fit(X, y)
        self.trained = True

    def predict(self, feature_vector: List[float]) -> Optional[int]:
        # Predict anomaly (1) or normal (0)
        if not self.trained:
            return None
        X = np.array(feature_vector).reshape(1, -1)
        return int(self.model.predict(X)[0])

