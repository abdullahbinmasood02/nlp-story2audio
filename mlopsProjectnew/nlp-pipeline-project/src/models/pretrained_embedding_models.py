import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
import mlflow
import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.utils.metrics import calculate_classification_metrics, Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class PretrainedEmbeddingClassifier:
    """Class for pretrained word embedding based text classification."""
    
    def __init__(self, embedding_type="glove-wiki-gigaword-100", classifier_type="logistic_regression", experiment_name="pretrained_embedding_models"):
        """
        Initialize the classifier.
        
        Args:
            embedding_type: Type of pretrained embeddings to use
            classifier_type: Type of classifier to use ("logistic_regression" or "svm")
            experiment_name: Name of MLflow experiment
        """
        self.embedding_type = embedding_type
        self.classifier_type = classifier_type
        self.experiment_name = experiment_name
        self.word_vectors = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
    
    def _load_embeddings(self):
        """
        Load pretrained word embeddings.
        
        Returns:
            Word vectors model
        """
        logger.info(f"Loading pretrained embeddings: {self.embedding_type}")
        
        try:
            # Cache path for embeddings
            cache_path = MODELS_DIR / f"{self.embedding_type}_vectors.kv"
            
            # Try to load from cache first
            if cache_path.exists():
                logger.info(f"Loading embeddings from cache: {cache_path}")
                self.word_vectors = KeyedVectors.load(str(cache_path))
            else:
                # Download and load embeddings
                self.word_vectors = gensim_downloader.load(self.embedding_type)
                
                # Save to cache
                logger.info(f"Saving embeddings to cache: {cache_path}")
                self.word_vectors.save(str(cache_path))
            
            logger.info(f"Loaded {len(self.word_vectors.index_to_key)} word vectors with dimension {self.word_vectors.vector_size}")
            
            return self.word_vectors
        
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def _text_to_vector(self, text, normalize=True):
        """
        Convert text to embedding vector by averaging word vectors.
        
        Args:
            text: Input text string
            normalize: Whether to normalize the vector
            
        Returns:
            Averaged word vector
        """
        if not self.word_vectors:
            self._load_embeddings()
        
        # Split text into words
        words = str(text).lower().split()
        
        # Get word vectors
        word_vecs = [self.word_vectors[word] for word in words if word in self.word_vectors]
        
        if not word_vecs:
            # Return zero vector if no words found
            return np.zeros(self.word_vectors.vector_size)
        
        # Average word vectors
        text_vector = np.mean(word_vecs, axis=0)
        
        # Normalize if requested
        if normalize:
            text_vector = text_vector / np.linalg.norm(text_vector)
        
        return text_vector
    
    def preprocess_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Preprocess text data for classification.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Preprocessed data
        """
        # Load embeddings if not already loaded
        if not self.word_vectors:
            self._load_embeddings()
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Convert texts to vectors
        logger.info("Converting training texts to vectors...")
        X_train_vecs = np.array([self._text_to_vector(text) for text in X_train])
        
        # Process validation data if provided
        X_val_vecs = None
        y_val_encoded = None
        if X_val is not None and y_val is not None:
            logger.info("Converting validation texts to vectors...")
            X_val_vecs = np.array([self._text_to_vector(text) for text in X_val])
            y_val_encoded = self.label_encoder.transform(y_val)
        
        return X_train_vecs, y_train_encoded, X_val_vecs, y_val_encoded
    
    def train(self, X_train, y_train, X_val=None, y_val=None, hyperparameter_tuning=False):
        """
        Train the classifier.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            hyperparameter_tuning: Whether to perform grid search
            
        Returns:
            Trained classifier
        """
        with mlflow.start_run(run_name=f"{self.embedding_type}_{self.classifier_type}"):
            # Log parameters
            mlflow.log_param("embedding_type", self.embedding_type)
            mlflow.log_param("classifier_type", self.classifier_type)
            mlflow.log_param("hyperparameter_tuning", hyperparameter_tuning)
            
            # Preprocess data
            X_train_vecs, y_train_encoded, X_val_vecs, y_val_encoded = self.preprocess_data(
                X_train, y_train, X_val, y_val
            )
            
            # Initialize classifier
            if self.classifier_type == "logistic_regression":
                self.classifier = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
                param_grid = {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                    "solver": ["liblinear", "lbfgs"]
                }
            elif self.classifier_type == "svm":
                self.classifier = LinearSVC(
                    random_state=42,
                    max_iter=10000
                )
                param_grid = {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                    "dual": [False],
                    "loss": ["squared_hinge"]
                }
            else:
                raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")
            
            # Train classifier
            if hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning...")
                
                # Log grid search parameters
                for param_name, param_values in param_grid.items():
                    mlflow.log_param(f"grid_{param_name}", str(param_values))
                
                # Create grid search
                grid_search = GridSearchCV(
                    self.classifier,
                    param_grid,
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=-1
                )
                
                # Train with grid search
                with Timer("Grid search time") as timer:
                    grid_search.fit(X_train_vecs, y_train_encoded)
                
                # Get best model
                self.classifier = grid_search.best_estimator_
                
                # Log best parameters
                for param, value in grid_search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)
                
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                mlflow.log_metric("grid_search_time", timer.interval)
            else:
                # Train without hyperparameter tuning
                logger.info("Training classifier...")
                with Timer("Training time") as timer:
                    self.classifier.fit(X_train_vecs, y_train_encoded)
                
                mlflow.log_metric("training_time", timer.interval)
            
            # Evaluate on training data
            train_preds = self.classifier.predict(X_train_vecs)
            train_metrics = calculate_classification_metrics(y_train_encoded, train_preds)
            
            # Log training metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            # Evaluate on validation data if provided
            if X_val_vecs is not None and y_val_encoded is not None:
                val_preds = self.classifier.predict(X_val_vecs)
                val_metrics = calculate_classification_metrics(y_val_encoded, val_preds)
                
                # Log validation metrics
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            return self.classifier
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Text data
        
        Returns:
            Predicted labels
        """
        if self.classifier is None:
            logger.error("Classifier not trained or loaded")
            return None
        
        # Convert texts to vectors
        X_vecs = np.array([self._text_to_vector(text) for text in X])
        
        # Make predictions
        pred_indices = self.classifier.predict(X_vecs)
        
        # Convert encoded predictions back to original labels
        return self.label_encoder.inverse_transform(pred_indices)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Text data
            
        Returns:
            Class probabilities
        """
        if self.classifier is None:
            logger.error("Classifier not trained or loaded")
            return None
        
        # Convert texts to vectors
        X_vecs = np.array([self._text_to_vector(text) for text in X])
        
        # Check if classifier has predict_proba
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_vecs)
        elif hasattr(self.classifier, "decision_function"):
            # For SVM, convert decision function to probabilities
            decision_values = self.classifier.decision_function(X_vecs)
            if decision_values.ndim == 1:
                # Binary classification
                probs = np.zeros((len(X), 2))
                probs[:, 1] = 1 / (1 + np.exp(-decision_values))
                probs[:, 0] = 1 - probs[:, 1]
                return probs
            else:
                # Multiclass classification
                # Use softmax to convert decision values to probabilities
                exp_scores = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            logger.error("Classifier does not support probability estimation")
            return None
    
    def save_model(self, model_path=None):
        """
        Save classifier and label encoder.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path to saved model
        """
        if self.classifier is None:
            logger.error("Classifier not trained or loaded")
            return None
        
        if model_path is None:
            model_path = MODELS_DIR / f"pretrained_embedding_{self.embedding_type}_{self.classifier_type}_model.pkl"
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save classifier and label encoder
        with open(model_path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "embedding_type": self.embedding_type
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(self.classifier, f"pretrained_embedding_{self.classifier_type}")
        
        # Log the embedding type
        mlflow.log_param("embedding_saved", self.embedding_type)
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load saved classifier and label encoder.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded classifier
        """
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            
            self.classifier = data["classifier"]
            self.label_encoder = data["label_encoder"]
            self.embedding_type = data["embedding_type"]
            
            # Make sure embeddings are loaded
            self._load_embeddings()
            
            logger.info(f"Model loaded from {model_path}")
            
            return self.classifier
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None