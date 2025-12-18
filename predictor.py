import os
import json
import joblib
import pandas as pd
import numpy as np
import gdown
import streamlit as st

class SEPredictor:
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.weights = {}
        
        # Create model directory if not exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download models if not exists
        self._download_models_if_needed()
        
        # Load necessary data
        self._load_metadata()
        self._load_models()
        self._calculate_weights()
    
    def _download_models_if_needed(self):
        """Google Driveからモデルをダウンロード"""
        
        # Google DriveのファイルIDを設定
        # 例: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        # からFILE_IDの部分を抽出
        
        model_urls = {
            'extratrees_model.pkl': '1dwdE3Y1F_KjLh7h7J4nvnHQJkjle8Dxf',
            'mlp_model.pkl': 'YOUR_MLP_MODEL_FILE_ID',
            'catboost_model.pkl': '1JMLlil7JRwvJ3xHwJr0AvRnrhr6qJ2t6',
            'extratrees_scaler.pkl': '',
            'mlp_scaler.pkl': '1UYSWG9C55gMahbutf3Gk2FYg4gf6kHfA',
            'catboost_scaler.pkl': 'YOUR_CATBOOST_SCALER_FILE_ID',
        }
        
        # Streamlitの進捗表示
        if 'streamlit' in dir():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        total_files = len(model_urls)
        
        for idx, (filename, file_id in enumerate(model_urls.items())):
            filepath = os.path.join(self.model_dir, filename)
            
            if not os.path.exists(filepath) and file_id != 'YOUR_..._FILE_ID':
                if 'streamlit' in dir():
                    status_text.text(f'Downloading {filename}...')
                
                try:
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, filepath, quiet=True)
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
            
            if 'streamlit' in dir():
                progress_bar.progress((idx + 1) / total_files)
        
        if 'streamlit' in dir():
            progress_bar.empty()
            status_text.empty()
                
    def _load_metadata(self):
        meta_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            print("Warning: metadata.json not found. Using default weights.")
                
    def _load_models(self):
        expected_models = ['mlp', 'extratrees', 'catboost']
        
        for name in expected_models:
            # Load model
            model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    print(f"Loaded {name} model")
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
            else:
                print(f"Model file not found: {model_path}")
            
            # Load scaler if exists
            scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    self.scalers[name] = joblib.load(scaler_path)
                    print(f"Loaded {name} scaler")
                except Exception as e:
                    print(f"Error loading {name} scaler: {e}")
                    
    def _calculate_weights(self):
        """Calculate weights based on CV R2 from metadata"""
        
        if not self.metadata or 'models' not in self.metadata:
            # Fallback to equal weights if no metadata
            n = len(self.models)
            if n > 0:
                for name in self.models:
                    self.weights[name] = 1.0 / n
            return

        r2_scores = {}
        for meta_name, info in self.metadata['models'].items():
            key = meta_name.lower()
            if key in self.models:
                r2 = info.get('performance', {}).get('outer_r2_mean', 0)
                r2_scores[key] = max(0, r2)
        
        total_r2 = sum(r2_scores.values())
        if total_r2 > 0:
            for name, r2 in r2_scores.items():
                self.weights[name] = r2 / total_r2
        else:
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
        
        # Normalize if some models failed
        if weight_sum > 0:
            ensemble_pred /= weight_sum
            
        return ensemble_pred, predictions
