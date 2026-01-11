from pydantic import BaseModel, Field, validator
import re

class TextFeatures(BaseModel):
    """Схема для признаков текста"""
    text_length: int = Field(..., description="Длина текста в символах", ge=1)
    num_words: int = Field(..., description="Количество слов", ge=1)
    num_unique_chars: int = Field(..., description="Количество уникальных символов", ge=1)
    
    @validator('text_length')
    def validate_text_length(cls, v):
        if v > 10000:
            raise ValueError('Текст слишком длинный')
        return v

class TextRequest(BaseModel):
    """Схема запроса"""
    text: str = Field(..., description="Текст для определения языка", min_length=3)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Текст не может быть пустым')
        return v

class PredictionResponse(BaseModel):
    """Схема ответа"""
    language: str = Field(..., description="Определенный язык")
    confidence: float = Field(..., description="Уверенность модели", ge=0, le=1)
    language_code: str = Field(..., description="Код языка")
    features: dict = Field(..., description="Извлеченные признаки")