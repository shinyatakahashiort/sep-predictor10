
import os
import json
import joblib
import pandas as pd
import numpy as np

class SEPredictor:
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.weights = {}
        
        # Load necessary data
        self._load_metadata()
        self._load_models()
        self._calculate_weights()
        
    def _load_metadata(self):
        meta_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            print("Warning: metadata.json not found.")
                
    def _load_models(self):
        # Names based on training script
        # Keys in metadata are usually capitalized (MLP, ExtraTrees, CatBoost)
        # Filenames are lowercase (mlp_model.pkl, etc.)
        expected_models = ['mlp', 'extratrees', 'catboost']
        
        for name in expected_models:
            # Load model
            model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
            
            # Load scaler if exists
            scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    self.scalers[name] = joblib.load(scaler_path)
                except Exception as e:
                    print(f"Error loading {name} scaler: {e}")
                    
    def _calculate_weights(self):
        # Calculate weights based on CV R2 from metadata
        # Metadata structure: models -> Name -> performance -> outer_r2_mean
        
        if not self.metadata or 'models' not in self.metadata:
            # Fallback to equal weights if no metadata
            n = len(self.models)
            if n > 0:
                for name in self.models:
                    self.weights[name] = 1.0 / n
            return

        r2_scores = {}
        for meta_name, info in self.metadata['models'].items():
            # Map metadata name (e.g. 'MLP') to model key (e.g. 'mlp')
            key = meta_name.lower()
            if key in self.models:
                r2 = info.get('performance', {}).get('outer_r2_mean', 0)
                r2_scores[key] = max(0, r2) # Ensure non-negative
        
        total_r2 = sum(r2_scores.values())
        if total_r2 > 0:
            for name, r2 in r2_scores.items():
                self.weights[name] = r2 / total_r2
        else:
            # Fallback if R2 sum is 0 or no matching models
            n = len(self.models)
            for name in self.models:
                self.weights[name] = 1.0 / n
                
    def predict(self, input_data):
        '''
        input_data: dict or DataFrame containing:
        '年齢', '性別', 'K', 'AL', 'LT', 'ACD'
        '''
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        # Ensure correct column order
        feature_cols = ['年齢', '性別', 'K', 'AL', 'LT', 'ACD']
        
        # Check if columns exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        df = df[feature_cols]
        
        predictions = {}
        
        for name, model in self.models.items():
            X = df.copy()
            # Apply scaling if a scaler exists for this model
            if name in self.scalers:
                X = self.scalers[name].transform(X)
            
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions[name] = np.zeros(len(df))
            
        # Weighted ensemble
        ensemble_pred = np.zeros(len(df))
        weight_sum = 0
        
        for name, pred in predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                ensemble_pred += weight * pred
                weight_sum += weight
        
        # Normalize if some models failed (though weight sum should be 1)
        if weight_sum > 0:
            ensemble_pred /= weight_sum
            
        return ensemble_pred, predictions
