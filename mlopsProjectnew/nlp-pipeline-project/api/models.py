import mlflow
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import os
import sys
import json
from pathlib import Path
import time
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Class to load and manage ML models from MLflow."""
    
    def __init__(self, tracking_uri="http://localhost:5000"):
        """
        Initialize ModelLoader.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.model = None
        self.text_column = "processed_text"
        self.preprocess_func = None
    
    def load_model_from_registry(self, model_name="news_category_classifier", stage="Production"):
        """
        Load model from MLflow model registry.
        
        Args:
            model_name: Name of registered model
            stage: Stage to load (Production, Staging, etc.)
            
        Returns:
            self with loaded model
        """
        try:
            logger.info(f"Loading {model_name} from registry (stage: {stage})")
            
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(model_name, stages=[stage])
            
            if not model_versions:
                raise ValueError(f"No {stage} versions found for model {model_name}")
            
            model_version = model_versions[0]
            logger.info(f"Found model version {model_version.version} with run_id {model_version.run_id}")
            
            # Load model
            model_uri = f"models:/{model_name}/{stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Get run details
            run = client.get_run(model_version.run_id)
            
            # Extract preprocessing function if available
            if "preprocess_func" in run.data.params:
                preprocess_name = run.data.params["preprocess_func"]
                # TODO: Implement loading of preprocessing function
            
            logger.info(f"Model loaded successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error loading model from registry: {str(e)}")
            raise
    
    def load_local_model(self, model_path):
        """
        Load model from local file.
        
        Args:
            model_path: Path to model pickle file
            
        Returns:
            self with loaded model
        """
        try:
            logger.info(f"Loading model from local path: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model from pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Model loaded successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            raise
    
    def preprocess_text(self, texts):
        """
        Apply preprocessing to input texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            Processed texts
        """
        if self.preprocess_func:
            return [self.preprocess_func(text) for text in texts]
        else:
            # Simple fallback preprocessing - lowercase and strip
            return [text.lower().strip() for text in texts]
    
    def predict(self, texts, top_k=3):
        """
        Make predictions with loaded model.
        
        Args:
            texts: List of texts to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of (category, probability) tuples for each text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_registry() or load_local_model() first.")
        
        try:
            # Preprocess text if needed
            processed_texts = self.preprocess_text(texts)
            
            # Create DataFrame if needed (some models expect DataFrame input)
            if hasattr(self.model, "predict_proba"):
                # For scikit-learn models
                if hasattr(self.model, "classes_"):
                    classes = self.model.classes_
                    probabilities = self.model.predict_proba(processed_texts)
                    
                    # For single text
                    if len(texts) == 1:
                        # Sort probabilities
                        sorted_indices = np.argsort(probabilities[0])[::-1][:top_k]
                        return [(classes[idx], probabilities[0][idx]) for idx in sorted_indices]
                    else:
                        # For multiple texts
                        results = []
                        for i, probs in enumerate(probabilities):
                            sorted_indices = np.argsort(probs)[::-1][:top_k]
                            results.append([(classes[idx], probs[idx]) for idx in sorted_indices])
                        return results[0]  # Return first result if only one text
                else:
                    # Generic model with predict_proba
                    predictions = self.model.predict(processed_texts)
                    probabilities = self.model.predict_proba(processed_texts)
                    
                    # Get top_k classes and probabilities
                    top_indices = np.argsort(probabilities[0])[::-1][:top_k]
                    return [(predictions[0], probabilities[0][i]) for i in top_indices]
            
            else:
                # For models without predict_proba
                predictions = self.model.predict(processed_texts)
                # Return predictions with 1.0 confidence (no probabilities available)
                return [(predictions[0], 1.0)]
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise