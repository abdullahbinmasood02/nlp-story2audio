import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro"))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro"))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    
    # Micro-averaged metrics (useful for imbalanced data)
    metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro"))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro"))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro"))
    
    # Weighted average
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    
    # Log metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(10, 8), cmap="Blues"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        figsize: Figure size tuple
        cmap: Colormap for heatmap
    
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, 
                xticklabels=labels, yticklabels=labels)
    
    # Set labels
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    return fig

def generate_classification_report(y_true, y_pred, labels=None):
    """
    Generate classification report as a DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        
    Returns:
        DataFrame with classification metrics per class
    """
    report = classification_report(
        y_true, 
        y_pred, 
        labels=labels, 
        output_dict=True
    )
    
    # Convert to DataFrame
    df_report = pd.DataFrame(report).transpose()
    
    return df_report

class Timer:
    """Utility class for timing code execution."""
    
    def __init__(self, name="Timer"):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.info(f"{self.name}: {self.interval:.4f} seconds")