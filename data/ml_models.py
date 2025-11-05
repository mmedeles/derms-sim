"""
ML model classifiers for anomaly detection.
"""

import pickle
from pathlib import Path
import numpy as np

class RF_Classifier:
    """Random Forest anomaly classifier using moving average deviations."""
    
    def __init__(self, model_path="models/rf_model_final.pkl"):
        """Load the trained Random Forest model."""
        model_file = Path(model_path)
        
        if not model_file.exists():
            print(f"[RF_Classifier] WARNING: Model file not found at {model_path}")
            print(f"[RF_Classifier] Using fallback threshold detector")
            self.model = None
            self.use_fallback = True
        else:
            try:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.use_fallback = False
                print(f"[RF_Classifier] Successfully loaded Random Forest from {model_path}")
                print(f"[RF_Classifier] Model expects 4 features: [dev_ma60, dev_ma120, dev_ma180, dev_ma240]")
                
            except Exception as e:
                print(f"[RF_Classifier] ERROR loading model: {e}")
                print(f"[RF_Classifier] Using fallback threshold detector")
                self.model = None
                self.use_fallback = True
    
    def classify(self, x) -> bool:
        """
        Classify a sample as anomaly (True) or normal (False).
        
        Args:
            x: List of [Hz_adjusted, ma60, ma120, ma180, ma240]
               First element is current Hz, rest are moving averages
        
        Returns:
            bool: True if anomaly, False if normal
        """
        # Fallback simple threshold detector if model didn't load
        if self.use_fallback:
            if len(x) >= 2:
                deviation = abs(x[0] - x[1])
                return deviation > 0.005
            return False
        
        try:
            # Extract values
            if len(x) < 5:
                print(f"[RF_Classifier] ERROR: Expected at least 5 values, got {len(x)}")
                return False
            
            hz_adjusted = float(x[0])
            ma60 = float(x[1])
            ma120 = float(x[2])
            ma180 = float(x[3])
            ma240 = float(x[4])
            
            deviations = [
                hz_adjusted - ma60,
                hz_adjusted - ma120,
                hz_adjusted - ma180,
                hz_adjusted - ma240,
            ]
            
            features = np.array(deviations).reshape(1, -1)
            
            # Returns 0 (normal) or 1 (anomaly)
            prediction = self.model.predict(features)[0]
            
            return bool(prediction)
            
        except Exception as e:
            print(f"[RF_Classifier] ERROR during classification: {e}")
            print(f"[RF_Classifier] Input was: {x}")
            if len(x) >= 2:
                deviation = abs(x[0] - x[1])
                return deviation > 0.005
            return False


#class XGB_Classifer:
#class LSTM_Classifier:
