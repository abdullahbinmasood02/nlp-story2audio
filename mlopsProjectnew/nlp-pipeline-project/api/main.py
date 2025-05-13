from fastapi import FastAPI, HTTPException, Depends, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import mlflow
import pandas as pd
import numpy as np
import time
import logging
import os
import sys
from datetime import datetime
import json
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Import local modules
from api.models import ModelLoader
from api.utils import TextAnalyzer, RequestTracker

# Define Prometheus metrics
REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "method"])
PREDICTIONS = Counter("model_predictions_total", "Total model predictions", ["category"])
LATENCIES = Histogram("request_latency_seconds", "Request latency in seconds", ["endpoint"])
TEXT_LENGTH = Histogram("input_text_length", "Length of input text", buckets=[10, 50, 100, 200, 500, 1000, 2000])
ERROR_COUNTER = Counter("api_errors_total", "Total API errors", ["endpoint", "error_type"])
PREDICTION_CONFIDENCE = Gauge("prediction_confidence", "Confidence of predictions")

# Define data models
class TextRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text to classify")
    top_k: Optional[int] = Field(3, ge=1, description="Number of top categories to return")

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to classify")
    top_k: Optional[int] = Field(3, ge=1, description="Number of top categories to return")

class PredictionResponse(BaseModel):
    category: str
    probability: float

class TextClassificationResponse(BaseModel):
    text: str
    predictions: List[PredictionResponse]
    processing_time: float

class BatchClassificationResponse(BaseModel):
    results: List[TextClassificationResponse]
    processing_time: float

class ModelInfo(BaseModel):
    name: str
    version: str
    metrics: Dict[str, float]
    creation_timestamp: str

# Initialize FastAPI app
app = FastAPI(
    title="News Category Classification API",
    description="API for classifying news articles into categories",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="api/templates")

# Model cache
MODEL_CACHE = {}

def get_model():
    """Get model from MLflow model registry."""
    try:
        # Get latest model from production stage
        client = mlflow.tracking.MlflowClient()
        latest_model = client.get_latest_versions("news_category_classifier", stages=["Production"])[0]
        model_uri = f"models:/news_category_classifier/{latest_model.version}"
        
        # Load the model
        logger.info(f"Loading model from {model_uri}")
        model_loader = ModelLoader(model_uri)
        MODEL_CACHE["model"] = model_loader
        MODEL_CACHE["info"] = {
            "name": "news_category_classifier",
            "version": latest_model.version,
            "metrics": latest_model.metrics,
            "creation_timestamp": latest_model.creation_timestamp,
        }
        logger.info(f"Model loaded: {MODEL_CACHE['info']}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Fallback to a model file if registry access fails
        model_path = os.path.join(os.path.dirname(__file__), "../models/fallback_model.pkl")
        model_loader = ModelLoader(model_path)
        MODEL_CACHE["model"] = model_loader
        MODEL_CACHE["info"] = {
            "name": "news_category_classifier",
            "version": "fallback",
            "metrics": {},
            "creation_timestamp": datetime.now().isoformat(),
        }

@app.get("/")
async def home(request: Request):
    """Render home page with API information."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=TextClassificationResponse)
async def predict(request: TextRequest):
    """
    Classify text into news categories.
    
    Returns the top k most likely categories and their probabilities.
    """
    start_time = time.time()
    REQUESTS.labels(endpoint="/predict", method="POST").inc()
    
    try:
        # Record input text length
        TEXT_LENGTH.observe(len(request.text))

        # Get model
        model_loader = get_model()

        # Make prediction
        predictions, probabilities = model_loader.predict_single(request.text, top_k=request.top_k)
        
        # Record prediction categories
        for category in predictions:
            PREDICTIONS.labels(category=category).inc()
        
        # Record confidence
        PREDICTION_CONFIDENCE.set(float(probabilities[0]))

        # Create response
        result = TextClassificationResponse(
            text=request.text,
            predictions=[
                PredictionResponse(category=category, probability=float(prob))
                for category, prob in zip(predictions, probabilities)
            ],
            processing_time=time.time() - start_time
        )
        
        # Record latency
        LATENCIES.labels(endpoint="/predict").observe(result.processing_time)
        
        return result
        
    except Exception as e:
        error_type = type(e).__name__
        ERROR_COUNTER.labels(endpoint="/predict", error_type=error_type).inc()
        logger.error(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchClassificationResponse)
async def batch_predict(request: BatchTextRequest):
    """
    Batch classify texts into news categories.
    """
    start_time = time.time()
    REQUESTS.labels(endpoint="/batch-predict", method="POST").inc()
    
    try:
        # Get model
        model_loader = get_model()

        # Process each text
        results = []
        for text in request.texts:
            # Record input text length
            TEXT_LENGTH.observe(len(text))
            
            # Make prediction
            text_start_time = time.time()
            predictions, probabilities = model_loader.predict_single(text, top_k=request.top_k)
            
            # Record prediction categories
            for category in predictions:
                PREDICTIONS.labels(category=category).inc()
            
            # Create response for this text
            results.append(
                TextClassificationResponse(
                    text=text,
                    predictions=[
                        PredictionResponse(category=category, probability=float(prob))
                        for category, prob in zip(predictions, probabilities)
                    ],
                    processing_time=time.time() - text_start_time
                )
            )
        
        # Create batch response
        batch_result = BatchClassificationResponse(
            results=results,
            processing_time=time.time() - start_time
        )
        
        # Record latency
        LATENCIES.labels(endpoint="/batch-predict").observe(batch_result.processing_time)
        
        return batch_result
        
    except Exception as e:
        error_type = type(e).__name__
        ERROR_COUNTER.labels(endpoint="/batch-predict", error_type=error_type).inc()
        logger.error(f"Error in /batch-predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """
    Get information about the currently loaded model.
    """
    REQUESTS.labels(endpoint="/model-info", method="GET").inc()
    
    try:
        # Get model
        _ = get_model()
        
        # Create response
        info = MODEL_CACHE["info"]
        model_info = ModelInfo(
            name=info["name"],
            version=info["version"],
            metrics=info["metrics"],
            creation_timestamp=info["creation_timestamp"]
        )
        
        return model_info
        
    except Exception as e:
        error_type = type(e).__name__
        ERROR_COUNTER.labels(endpoint="/model-info", error_type=error_type).inc()
        logger.error(f"Error in /model-info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    REQUESTS.labels(endpoint="/health", method="GET").inc()
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)