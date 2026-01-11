import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import re

class LanguageDetector:
    def __init__(self, model_path: str = "best_model.joblib", scaler_path: str = "scaler.joblib"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Соответствие кодов языков (зависит от вашей кодировки в ЛР4)
        self.language_mapping = {
            0: {"name": "English", "code": "en"},
            1: {"name": "Russian", "code": "ru"},
            2: {"name": "Spanish", "code": "es"},
            # Добавьте остальные языки из вашего датасета
        }
    
    def extract_features(self, text: str) -> Dict:
        """Извлечение признаков из текста"""
        text_length = len(text)
        num_words = len(text.split())
        num_unique_chars = len(set(text.lower()))
        
        return {
            'text_length': text_length,
            'num_words': num_words,
            'num_unique_chars': num_unique_chars
        }
    
    def predict(self, text: str) -> Tuple[str, float, str]:
        """Предсказание языка"""
        # Извлекаем признаки
        features = self.extract_features(text)
        
        # Преобразуем в DataFrame
        features_df = pd.DataFrame([features])
        
        # Масштабируем
        features_scaled = self.scaler.transform(features_df)
        
        # Предсказываем
        prediction = self.model.predict(features_scaled)[0]
        confidence = max(self.model.predict_proba(features_scaled)[0])
        
        # Получаем информацию о языке
        lang_info = self.language_mapping.get(prediction, {"name": "Unknown", "code": "unk"})
        
        return lang_info["name"], confidence, lang_info["code"], features