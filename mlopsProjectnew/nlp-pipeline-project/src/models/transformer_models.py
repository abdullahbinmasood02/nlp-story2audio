import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
import mlflow
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
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

class NewsDataset(Dataset):
    """PyTorch Dataset for news category data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels (encoded)
            tokenizer: Tokenizer for encoding the texts
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to tensors
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

class TransformerClassifier:
    """Class for transformer-based text classification."""
    
    def __init__(self, model_name="distilbert-base-uncased", experiment_name="transformer_models"):
        """
        Initialize the transformer classifier.
        
        Args:
            model_name: Name of the transformer model to use
            experiment_name: Name of MLflow experiment
        """
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.batch_size = 16
        self.max_length = 128
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Using device: {self.device}")
    
    def preprocess_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Preprocess data for training.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Processed data loaders
        """
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Encode labels
        self.label_encoder.fit(y_train)
        num_labels = len(self.label_encoder.classes_)
        logger.info(f"Number of classes: {num_labels}")
        
        y_train_encoded = self.label_encoder.transform(y_train)
        
        # Create training dataset and dataloader
        train_dataset = NewsDataset(
            texts=X_train,
            labels=y_train_encoded,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create validation dataset and dataloader if validation data is provided
        val_dataloader = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            
            val_dataset = NewsDataset(
                texts=X_val,
                labels=y_val_encoded,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        return train_dataloader, val_dataloader, num_labels
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=3, learning_rate=5e-5):
        """
        Train the transformer model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        with mlflow.start_run(run_name=f"{self.model_name}_classifier"):
            # Log parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_length", self.max_length)
            
            # Preprocess data
            train_dataloader, val_dataloader, num_labels = self.preprocess_data(
                X_train, y_train, X_val, y_val
            )
            
            # Load pretrained model
            logger.info(f"Loading pretrained model: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            ).to(self.device)
            
            # Prepare optimizer and scheduler
            optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                eps=1e-8
            )
            
            # Calculate total training steps
            total_steps = len(train_dataloader) * epochs
            
            # Create learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Training loop
            logger.info("Starting training...")
            
            train_losses = []
            val_losses = []
            
            with Timer("Training time") as timer:
                for epoch in range(epochs):
                    logger.info(f"Epoch {epoch+1}/{epochs}")
                    
                    # Training
                    self.model.train()
                    train_loss = 0
                    
                    for batch in tqdm(train_dataloader, desc="Training"):
                        # Move batch to device
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["label"].to(self.device)
                        
                        # Reset gradients
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        train_loss += loss.item()
                        
                        # Backward pass
                        loss.backward()
                        
                        # Update weights
                        optimizer.step()
                        scheduler.step()
                    
                    # Calculate average training loss
                    avg_train_loss = train_loss / len(train_dataloader)
                    train_losses.append(avg_train_loss)
                    
                    logger.info(f"Average training loss: {avg_train_loss:.4f}")
                    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                    
                    # Validation
                    if val_dataloader is not None:
                        val_loss, val_acc, val_f1 = self._evaluate(val_dataloader)
                        val_losses.append(val_loss)
                        
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        logger.info(f"Validation accuracy: {val_acc:.4f}")
                        logger.info(f"Validation F1 score: {val_f1:.4f}")
                        
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                        mlflow.log_metric("val_f1_macro", val_f1, step=epoch)
            
            # Log execution time
            mlflow.log_metric("training_time_seconds", timer.interval)
            
            # Final evaluation
            if val_dataloader is not None:
                val_loss, val_acc, val_f1 = self._evaluate(val_dataloader)
                
                logger.info("Final evaluation:")
                logger.info(f"Validation loss: {val_loss:.4f}")
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                logger.info(f"Validation F1 score: {val_f1:.4f}")
                
                mlflow.log_metric("final_val_loss", val_loss)
                mlflow.log_metric("final_val_accuracy", val_acc)
                mlflow.log_metric("final_val_f1_macro", val_f1)
            
            # Save model architecture summary
            model_summary = str(self.model)
            mlflow.log_text(model_summary, "model_summary.txt")
            
            # Save model with MLflow
            mlflow.pytorch.log_model(self.model, "pytorch_model")
            
            return self.model
    
    def _evaluate(self, dataloader):
        """
        Evaluate the model on a dataloader.
        
        Args:
            dataloader: DataLoader with validation/test data
            
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Store predictions and labels
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return val_loss / len(dataloader), accuracy, f1
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Text data (list or pandas Series)
            
        Returns:
            Predicted labels and probabilities
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None, None
        
        # Convert to list if necessary
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, str):
            X = [X]
        
        # Create dataset
        dataset = NewsDataset(
            texts=X,
            labels=[0] * len(X),  # Dummy labels
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get predictions
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get probabilities
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        
        # Get predicted labels and probabilities
        all_probs = np.array(all_probs)
        pred_indices = np.argmax(all_probs, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_indices)
        
        return pred_labels, all_probs
    
    def save_model(self, output_dir=None):
        """
        Save model, tokenizer, and label encoder.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        if output_dir is None:
            output_dir = MODELS_DIR / f"transformer_{self.model_name.split('/')[-1]}"
        
        # Create directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        np.save(output_dir / "label_encoder_classes.npy", self.label_encoder.classes_)
        
        logger.info(f"Model saved to {output_dir}")
        
        return output_dir
    
    def load_model(self, model_dir):
        """
        Load saved model, tokenizer, and label encoder.
        
        Args:
            model_dir: Directory with saved model
            
        Returns:
            Loaded model
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            logger.error(f"Model directory {model_dir} does not exist")
            return None
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            
            # Load label encoder
            label_encoder_path = model_dir / "label_encoder_classes.npy"
            if label_encoder_path.exists():
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
            
            logger.info(f"Model loaded from {model_dir}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None