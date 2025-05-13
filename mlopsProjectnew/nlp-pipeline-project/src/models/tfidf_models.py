import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
import pickle
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.utils.metrics import calculate_classification_metrics, Timer, plot_confusion_matrix

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

class TfidfClassifier:
    """Class for TF-IDF based text classification models."""
    
    def __init__(self, experiment_name="news_category_classification"):
        """
        Initialize TF-IDF classifier.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Initialized TfidfClassifier with experiment: {experiment_name}")
    
    def load_data(self):
        """
        Load the train, validation, and test datasets.
        
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        try:
            train_path = PROCESSED_DATA_DIR / "train.csv"
            valid_path = PROCESSED_DATA_DIR / "validation.csv"
            test_path = PROCESSED_DATA_DIR / "test.csv"
            
            logger.info(f"Loading data from {PROCESSED_DATA_DIR}")
            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Loaded train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}")
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_naive_bayes(self, train_df, valid_df, text_column="processed_text", target_column="category"):
        """
        Train a Naive Bayes classifier with TF-IDF features.
        
        Args:
            train_df: Training DataFrame
            valid_df: Validation DataFrame
            text_column: Column containing processed text
            target_column: Column containing target labels
            
        Returns:
            Trained pipeline
        """
        logger.info("Training Naive Bayes with TF-IDF features")
        
        # Start MLflow run
        with mlflow.start_run(run_name="tfidf_naive_bayes"):
            timer = Timer()
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', MultinomialNB())
            ])
            
            # Set parameters for grid search
            param_grid = {
                'tfidf__max_features': [5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__alpha': [0.1, 0.5, 1.0]
            }
            
            # Create grid search
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=3, 
                scoring='f1_weighted', 
                verbose=1, 
                n_jobs=-1
            )
            
            # Fit grid search
            with timer:
                grid_search.fit(train_df[text_column], train_df[target_column])
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            
            # Evaluate on validation set
            y_pred = best_model.predict(valid_df[text_column])
            y_prob = best_model.predict_proba(valid_df[text_column])
            
            # Calculate metrics
            metrics = calculate_classification_metrics(valid_df[target_column], y_pred, y_prob)
            
            # Plot confusion matrix
            labels = sorted(valid_df[target_column].unique())
            cm_fig = plot_confusion_matrix(valid_df[target_column], y_pred, labels)
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'TF-IDF + Naive Bayes',
                'text_column': text_column,
                'target_column': target_column,
                'train_samples': len(train_df),
                'validation_samples': len(valid_df),
                **best_params
            })
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'training_time': timer.elapsed_time
            })
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log confusion matrix figure
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            
            # Save model locally
            model_path = MODELS_DIR / "tfidf_naive_bayes.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Model saved to {model_path}")
            
            return best_model
    
    def train_logistic_regression(self, train_df, valid_df, text_column="processed_text", target_column="category"):
        """
        Train a Logistic Regression classifier with TF-IDF features.
        
        Args:
            train_df: Training DataFrame
            valid_df: Validation DataFrame
            text_column: Column containing processed text
            target_column: Column containing target labels
            
        Returns:
            Trained pipeline
        """
        logger.info("Training Logistic Regression with TF-IDF features")
        
        # Start MLflow run
        with mlflow.start_run(run_name="tfidf_logistic_regression"):
            timer = Timer()
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ])
            
            # Set parameters for grid search
            param_grid = {
                'tfidf__max_features': [10000, 20000],
                'tfidf__ngram_range': [(1, 2), (1, 3)],
                'clf__C': [0.1, 1.0, 10.0]
            }
            
            # Create grid search
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=3, 
                scoring='f1_weighted', 
                verbose=1, 
                n_jobs=-1
            )
            
            # Fit grid search
            with timer:
                grid_search.fit(train_df[text_column], train_df[target_column])
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            
            # Evaluate on validation set
            y_pred = best_model.predict(valid_df[text_column])
            y_prob = best_model.predict_proba(valid_df[text_column])
            
            # Calculate metrics
            metrics = calculate_classification_metrics(valid_df[target_column], y_pred, y_prob)
            
            # Plot confusion matrix
            labels = sorted(valid_df[target_column].unique())
            cm_fig = plot_confusion_matrix(valid_df[target_column], y_pred, labels)
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'TF-IDF + Logistic Regression',
                'text_column': text_column,
                'target_column': target_column,
                'train_samples': len(train_df),
                'validation_samples': len(valid_df),
                **best_params
            })
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'training_time': timer.elapsed_time
            })
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log confusion matrix figure
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            
            # Save model locally
            model_path = MODELS_DIR / "tfidf_logistic_regression.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Model saved to {model_path}")
            
            return best_model
    
    def train_svm(self, train_df, valid_df, text_column="processed_text", target_column="category"):
        """
        Train an SVM classifier with TF-IDF features.
        
        Args:
            train_df: Training DataFrame
            valid_df: Validation DataFrame
            text_column: Column containing processed text
            target_column: Column containing target labels
            
        Returns:
            Trained pipeline
        """
        logger.info("Training SVM with TF-IDF features")
        
        # Start MLflow run
        with mlflow.start_run(run_name="tfidf_svm"):
            timer = Timer()
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', LinearSVC(random_state=42, max_iter=10000))
            ])
            
            # Set parameters for grid search
            param_grid = {
                'tfidf__max_features': [10000, 20000],
                'tfidf__ngram_range': [(1, 2)],
                'clf__C': [0.1, 1.0]
            }
            
            # Create grid search
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=3, 
                scoring='f1_weighted', 
                verbose=1, 
                n_jobs=-1
            )
            
            # Fit grid search
            with timer:
                grid_search.fit(train_df[text_column], train_df[target_column])
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            
            # Evaluate on validation set
            y_pred = best_model.predict(valid_df[text_column])
            
            # SVM doesn't have predict_proba, so pass None
            metrics = calculate_classification_metrics(valid_df[target_column], y_pred)
            
            # Plot confusion matrix
            labels = sorted(valid_df[target_column].unique())
            cm_fig = plot_confusion_matrix(valid_df[target_column], y_pred, labels)
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'TF-IDF + SVM',
                'text_column': text_column,
                'target_column': target_column,
                'train_samples': len(train_df),
                'validation_samples': len(valid_df),
                **best_params
            })
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'training_time': timer.elapsed_time
            })
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log confusion matrix figure
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            
            # Save model locally
            model_path = MODELS_DIR / "tfidf_svm.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Model saved to {model_path}")
            
            return best_model
    
    def evaluate_on_test_set(self, model, test_df, text_column="processed_text", target_column="category", model_name="tfidf_model"):
        """
        Evaluate a trained model on the test set.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            text_column: Column containing processed text
            target_column: Column containing target labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name} on test set")
        
        # Get predictions
        y_pred = model.predict(test_df[text_column])
        
        # Get probabilities if available
        try:
            y_prob = model.predict_proba(test_df[text_column])
        except:
            y_prob = None
        
        # Calculate metrics
        metrics = calculate_classification_metrics(test_df[target_column], y_pred, y_prob)
        
        # Log results
        logger.info(f"Test set results for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
        
        return metrics

if __name__ == "__main__":
    # Create classifier
    classifier = TfidfClassifier()
    
    # Load data
    train_df, valid_df, test_df = classifier.load_data()
    
    # Train models
    nb_model = classifier.train_naive_bayes(train_df, valid_df)
    lr_model = classifier.train_logistic_regression(train_df, valid_df)
    svm_model = classifier.train_svm(train_df, valid_df)
    
    # Evaluate on test set
    nb_metrics = classifier.evaluate_on_test_set(nb_model, test_df, model_name="Naive Bayes")
    lr_metrics = classifier.evaluate_on_test_set(lr_model, test_df, model_name="Logistic Regression")
    svm_metrics = classifier.evaluate_on_test_set(svm_model, test_df, model_name="SVM")