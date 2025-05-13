import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
import mlflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, GlobalMaxPooling1D
from sklearn.preprocessing import LabelEncoder
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

# Set TensorFlow log level
tf.get_logger().setLevel('ERROR')

class WordEmbeddingClassifier:
    """Class for word embedding based neural network text classification."""
    
    def __init__(self, model_type="simple_lstm", experiment_name="word_embedding_models"):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model to use ("simple_lstm", "bidirectional_lstm", or "cnn")
            experiment_name: Name of MLflow experiment
        """
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.label_encoder = LabelEncoder()
        self.tokenizer = None
        self.model = None
        self.max_sequence_length = 100
        self.max_words = 10000
        self.embedding_dim = 100
        
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
    
    def preprocess_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Preprocess text data for neural network.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Preprocessed data
        """
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create tokenizer
        logger.info("Fitting tokenizer...")
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert text to sequences
        X_train_sequences = self.tokenizer.texts_to_sequences(X_train)
        
        # Pad sequences
        X_train_padded = pad_sequences(X_train_sequences, maxlen=self.max_sequence_length)
        
        # Process validation data if provided
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_sequences = self.tokenizer.texts_to_sequences(X_val)
            X_val_padded = pad_sequences(X_val_sequences, maxlen=self.max_sequence_length)
            return X_train_padded, y_train_encoded, X_val_padded, y_val_encoded
        
        return X_train_padded, y_train_encoded
    
    def _build_model(self, num_classes):
        """
        Build neural network model.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        if self.model_type == "simple_lstm":
            model = Sequential([
                Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, 
                          input_length=self.max_sequence_length),
                LSTM(128),
                Dropout(0.2),
                Dense(num_classes, activation='softmax')
            ])
        elif self.model_type == "bidirectional_lstm":
            model = Sequential([
                Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, 
                          input_length=self.max_sequence_length),
                tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(LSTM(32)),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        elif self.model_type == "cnn":
            model = Sequential([
                Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, 
                          input_length=self.max_sequence_length),
                tf.keras.layers.Conv1D(128, 5, activation='relu'),
                tf.keras.layers.MaxPooling1D(5),
                tf.keras.layers.Conv1D(128, 5, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=5):
        """
        Train the model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        with mlflow.start_run(run_name=f"{self.model_type}_embedding"):
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("max_words", self.max_words)
            mlflow.log_param("embedding_dim", self.embedding_dim)
            mlflow.log_param("max_sequence_length", self.max_sequence_length)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            
            # Preprocess data
            if X_val is not None and y_val is not None:
                X_train_padded, y_train_encoded, X_val_padded, y_val_encoded = self.preprocess_data(
                    X_train, y_train, X_val, y_val
                )
                validation_data = (X_val_padded, y_val_encoded)
            else:
                X_train_padded, y_train_encoded = self.preprocess_data(X_train, y_train)
                validation_data = None
            
            # Get number of classes
            num_classes = len(set(y_train_encoded))
            mlflow.log_param("num_classes", num_classes)
            
            # Build model
            self.model = self._build_model(num_classes)
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=3, restore_best_weights=True
                )
            ]
            
            # Train model
            with Timer("Training time"):
                history = self.model.fit(
                    X_train_padded, y_train_encoded,
                    validation_data=validation_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Log training metrics
            for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("train_accuracy", acc, step=epoch)
            
            # Log validation metrics if available
            if validation_data is not None:
                for epoch, (val_loss, val_acc) in enumerate(
                    zip(history.history['val_loss'], history.history['val_accuracy'])
                ):
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                # Get final validation predictions for additional metrics
                val_preds = np.argmax(self.model.predict(X_val_padded), axis=1)
                val_metrics = calculate_classification_metrics(
                    y_val_encoded, val_preds
                )
                
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            # Log model architecture
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            mlflow.log_text("\n".join(model_summary), "model_summary.txt")
            
            # Save model architecture as figure
            tf.keras.utils.plot_model(
                self.model, 
                to_file='model_architecture.png',
                show_shapes=True
            )
            mlflow.log_artifact("model_architecture.png")
            
            return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Text data
            
        Returns:
            Predicted labels
        """
        # Convert texts to sequences
        X_sequences = self.tokenizer.texts_to_sequences(X)
        
        # Pad sequences
        X_padded = pad_sequences(X_sequences, maxlen=self.max_sequence_length)
        
        # Make predictions
        y_pred_probs = self.model.predict(X_padded)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Convert encoded predictions back to original labels
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Text data
            
        Returns:
            Class probabilities
        """
        X_sequences = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(X_sequences, maxlen=self.max_sequence_length)
        return self.model.predict(X_padded)
    
    def save_model(self, model_path=None, tokenizer_path=None, encoder_path=None):
        """
        Save the model and its components.
        
        Args:
            model_path: Path to save the model
            tokenizer_path: Path to save the tokenizer
            encoder_path: Path to save the label encoder
            
        Returns:
            Dictionary of saved paths
        """
        if model_path is None:
            model_path = MODELS_DIR / f"word_embedding_{self.model_type}_model"
        
        if tokenizer_path is None:
            tokenizer_path = MODELS_DIR / f"word_embedding_{self.model_type}_tokenizer.json"
        
        if encoder_path is None:
            encoder_path = MODELS_DIR / f"word_embedding_{self.model_type}_encoder.npy"
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save tokenizer
        import json
        tokenizer_json = self.tokenizer.to_json()
        with open(tokenizer_path, 'w') as f:
            f.write(tokenizer_json)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Save label encoder classes
        np.save(encoder_path, self.label_encoder.classes_)
        logger.info(f"Label encoder saved to {encoder_path}")
        
        # Log model to MLflow
        mlflow.keras.log_model(self.model, f"word_embedding_{self.model_type}")
        
        # Log artifacts
        mlflow.log_artifact(str(tokenizer_path))
        mlflow.log_artifact(str(encoder_path))
        
        return {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "encoder_path": encoder_path
        }
    
    def load_model(self, model_path, tokenizer_path, encoder_path):
        """
        Load a saved model and its components.
        
        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to the saved tokenizer
            encoder_path: Path to the saved label encoder
            
        Returns:
            Loaded model
        """
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load tokenizer
        import json
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = f.read()
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        logger.info(f"Tokenizer loaded from {tokenizer_path}")
        
        # Load label encoder classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(encoder_path, allow_pickle=True)
        logger.info(f"Label encoder loaded from {encoder_path}")
        
        return self.model