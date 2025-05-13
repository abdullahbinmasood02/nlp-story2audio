import numpy as np
import pandas as pd
import logging
from textblob import TextBlob
import nltk
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for text feature engineering."""
    
    def __init__(self):
        pass
    
    def extract_basic_features(self, text):
        """
        Extract basic text features.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of features
        """
        if not text or not isinstance(text, str):
            return {
                "text_length": 0,
                "word_count": 0,
                "avg_word_length": 0,
                "char_count": 0
            }
        
        # Split into words
        words = text.split()
        
        # Calculate features
        features = {
            "text_length": len(text),
            "word_count": len(words),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "char_count": sum(len(w) for w in words)
        }
        
        return features
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment features using TextBlob.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of sentiment features
        """
        if not text or not isinstance(text, str):
            return {
                "polarity": 0,
                "subjectivity": 0
            }
        
        # Calculate sentiment
        try:
            blob = TextBlob(text)
            return {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return {
                "polarity": 0,
                "subjectivity": 0
            }
    
    def extract_pos_features(self, text):
        """
        Extract part-of-speech features.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of POS features
        """
        if not text or not isinstance(text, str):
            return {
                "noun_count": 0,
                "verb_count": 0,
                "adj_count": 0,
                "adv_count": 0
            }
        
        try:
            # POS tagging
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # Count different POS
            pos_counts = Counter([tag[1] for tag in pos_tags])
            
            return {
                "noun_count": sum(pos_counts[tag] for tag in pos_counts if tag.startswith('NN')),
                "verb_count": sum(pos_counts[tag] for tag in pos_counts if tag.startswith('VB')),
                "adj_count": sum(pos_counts[tag] for tag in pos_counts if tag.startswith('JJ')),
                "adv_count": sum(pos_counts[tag] for tag in pos_counts if tag.startswith('RB'))
            }
        except Exception as e:
            logger.error(f"Error extracting POS features: {str(e)}")
            return {
                "noun_count": 0,
                "verb_count": 0,
                "adj_count": 0,
                "adv_count": 0
            }
    
    def extract_all_features(self, text):
        """
        Extract all text features.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of all features
        """
        # Combine all feature sets
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_pos_features(text))
        
        return features
    
    def feature_engineering_dataframe(self, df, text_column):
        """
        Add engineered features to a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            DataFrame with additional feature columns
        """
        logger.info(f"Extracting features from {text_column}...")
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Extract features for each text
        features_list = result_df[text_column].apply(self.extract_all_features).tolist()
        
        # Convert list of dictionaries to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Concatenate with original DataFrame
        result_df = pd.concat([result_df, features_df], axis=1)
        
        logger.info(f"Added {features_df.shape[1]} features")
        
        return result_df