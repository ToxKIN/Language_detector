from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import json
from typing import Dict, Any
import os

from app.database import get_db, PredictionLog
from app.schemas import TextRequest, PredictionResponse
from app.ml_utils import LanguageDetector
import datetime

app = FastAPI(title="Language Detection Service", version="3.0", docs_url="/api/docs")

# Подключаем статические файлы (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация детектора
detector = None

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте"""
    global detector
    detector = LanguageDetector("app/best_model.joblib", "app/scaler.joblib")
    print("✅ Модель и скейлер загружены")

# Главная страница с веб-интерфейсом
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": detector is not None}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_language(
    request: TextRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Определение языка текста
    
    - **text**: Текст для анализа (минимум 3 символа)
    """
    try:
        # Получаем предсказание
        language, confidence, language_code, features = detector.predict(request.text)
        
        # Подготавливаем ответ
        response_data = {
            "language": language,
            "confidence": round(float(confidence), 4),
            "language_code": language_code,
            "features": features
        }
        
        # Логируем в базу данных
        log_entry = PredictionLog(
            request_data=json.dumps({"text": request.text}),
            response_data=json.dumps(response_data),
            language=language,
            confidence=f"{confidence:.2%}"
        )
        db.add(log_entry)
        db.commit()
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/v1/logs")
async def get_logs(db: Session = Depends(get_db), limit: int = 10):
    """Получение последних записей логов"""
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    return {
        "total": len(logs),
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "language": log.language,
                "confidence": log.confidence
            }
            for log in logs
        ]
    }