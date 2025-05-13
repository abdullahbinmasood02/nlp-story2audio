import logging
import time
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import langid
import spacy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Try to load spaCy model for advanced text analysis
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Could not load spaCy model: {e}. Some features will be disabled.")
    nlp = None

class TextAnalyzer:
    """Utility class for text analysis."""
    
    def __init__(self):
        """Initialize TextAnalyzer."""
        self.stopwords = set(stopwords.words('english'))
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and extract features.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of text features
        """
        # Check for empty or invalid input
        if not text or not isinstance(text, str):
            return {
                "length": 0,
                "word_count": 0,
                "language": "unknown",
                "sentiment": 0,
                "subjective": 0,
            }
        
        # Basic metrics
        words = word_tokenize(text)
        word_count = len(words)
        
        # Language detection
        try:
            language = detect(text)
        except LangDetectException:
            language = "unknown"
        
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            "word_count": word_count,
            "language": language,
            "sentiment": sentiment,
            "subjectivity": subjectivity,
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing stopwords, punctuation, etc.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words if word.isalpha() and word not in self.stopwords]
        
        # Join back into string
        return " ".join(words)
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            str: Detected language code
        """
        try:
            lang, _ = langid.classify(text)
            return lang
        except Exception:
            return "unknown"

class RequestTracker:
    """Utility class for tracking API requests."""
    
    def __init__(self, log_file="api_requests.log"):
        """
        Initialize RequestTracker.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.log_path = Path(log_file)
        self.start_time = time.time()
        self.logs = []
        
        # Create directory if it doesn't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_request(self, endpoint: str, input_data: Any, output_data: Any, duration: float):
        """
        Log an API request.
        
        Args:
            endpoint: API endpoint
            input_data: Request input data
            output_data: Response output data
            duration: Request duration in seconds
        """
        log_entry = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "endpoint": endpoint,
            "input": input_data if isinstance(input_data, str) else str(input_data)[:200],
            "output": str(output_data)[:200],
            "duration": duration
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error logging request: {str(e)}")
    
    def log(self, message: str) -> None:
        """
        Log a message with timestamp.
        
        Args:
            message (str): Message to log
        """
        elapsed = time.time() - self.start_time
        self.logs.append((elapsed, message))
        
    def get_elapsed_time(self) -> float:
        """
        Get total elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.start_time
        
    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get formatted logs.
        
        Returns:
            List[Dict[str, Any]]: Formatted logs
        """
        return [{"time": time, "message": msg} for time, msg in self.logs]
        
    def get_recent_requests(self, n=10) -> List[Dict[str, Any]]:
        """
        Get recent API requests.
        
        Args:
            n: Number of requests to retrieve
            
        Returns:
            List of recent request logs
        """
        try:
            if not self.log_path.exists():
                return []
            
            with open(self.log_file, "r") as f:
                lines = f.readlines()
            
            # Parse JSON lines
            logs = [json.loads(line) for line in lines[-n:]]
            return logs
        
        except Exception as e:
            logger.error(f"Error retrieving request logs: {str(e)}")
            return []
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get request statistics.
        
        Returns:
            Dictionary with request statistics
        """
        try:
            if not self.log_path.exists():
                return {
                    "total_requests": 0,
                    "avg_duration": 0,
                    "endpoints": {}
                }
            
            with open(self.log_file, "r") as f:
                lines = f.readlines()
            
            # Parse JSON lines
            logs = [json.loads(line) for line in lines]
            
            # Calculate statistics
            total = len(logs)
            durations = [log["duration"] for log in logs]
            avg_duration = sum(durations) / total if total > 0 else 0
            
            # Count requests by endpoint
            endpoints = {}
            for log in logs:
                endpoint = log["endpoint"]
                if endpoint not in endpoints:
                    endpoints[endpoint] = 0
                endpoints[endpoint] += 1
            
            return {
                "total_requests": total,
                "avg_duration": avg_duration,
                "endpoints": endpoints
            }
        
        except Exception as e:
            logger.error(f"Error calculating request statistics: {str(e)}")
            return {
                "total_requests": 0,
                "avg_duration": 0,
                "endpoints": {}
            }