import pytest
import pandas as pd
import numpy as np
import sys
import os
import re
from pathlib import Path

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.text_cleaning import TextCleaner
from src.features.feature_engineering import FeatureEngineer
from src.data.preprocess import NewsDataPreprocessor

class TestTextCleaning:
    """Test cases for text cleaning functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
        self.test_text = "This is an Example text with Punctuation!!! 123 and URL: http://example.com"
    
    def test_basic_clean(self):
        """Test basic text cleaning."""
        cleaned = self.cleaner.basic_clean(self.test_text)
        
        # Check lowercase
        assert cleaned == cleaned.lower()
        
        # Check punctuation removed
        assert "!" not in cleaned
        assert ":" not in cleaned
        
        # Check numbers removed
        assert "123" not in cleaned
        
        # Check URL removed
        assert "http" not in cleaned
        assert "example.com" not in cleaned
        
        # Check content still exists
        assert "example text" in cleaned
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        text = "This is a test with some stopwords like the and a"
        cleaned = self.cleaner.remove_stopwords_from_text(text)
        
        # Check stopwords removed
        assert "this" not in cleaned.lower().split()
        assert "is" not in cleaned.lower().split()
        assert "a" not in cleaned.lower().split()
        assert "the" not in cleaned.lower().split()
        
        # Check content words still exist
        assert "test" in cleaned.lower().split()
        assert "stopwords" in cleaned.lower().split()
    
    def test_lemmatize_text(self):
        """Test lemmatization."""
        text = "The cats are running quickly through forests"
        lemmatized = self.cleaner.lemmatize_text(text)
        
        # Check lemmatization
        assert "cat" in lemmatized.split()
        assert "running" not in lemmatized.split()
        assert "run" in lemmatized.split()
        assert "forest" in lemmatized.split()
        assert "forests" not in lemmatized.split()
    
    def test_clean_text_full_pipeline(self):
        """Test full text cleaning pipeline."""
        original = "The CATS are running QUICKLY!!! through http://example.com forests. 123"
        processed = self.cleaner.clean_text(original)
        
        # Check results of full pipeline
        assert "cat" in processed.split()
        assert "run" in processed.split()
        assert "forest" in processed.split()
        assert "the" not in processed.split()
        assert "quickly" not in processed.split()  # Should be removed as stopword
        assert "http" not in processed
        assert "123" not in processed

class TestFeatureEngineering:
    """Test cases for feature engineering functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        self.test_text = "This is a sample text for feature extraction. It contains some sentiment words like happy and great!"
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        features = self.engineer.extract_basic_features(self.test_text)
        
        # Check basic features
        assert isinstance(features, dict)
        assert "text_length" in features
        assert features["text_length"] > 0
        assert features["text_length"] == len(self.test_text)
        assert features["word_count"] > 0
        assert features["avg_word_length"] > 0
    
    def test_extract_sentiment_features(self):
        """Test sentiment feature extraction."""
        # Positive text
        pos_text = "I am very happy and satisfied with the excellent results!"
        pos_features = self.engineer.extract_sentiment_features(pos_text)
        
        # Negative text
        neg_text = "I am disappointed and frustrated with the terrible outcome."
        neg_features = self.engineer.extract_sentiment_features(neg_text)
        
        # Check sentiment features
        assert isinstance(pos_features, dict)
        assert "polarity" in pos_features
        assert "subjectivity" in pos_features
        
        # Positive text should have higher polarity than negative
        assert pos_features["polarity"] > neg_features["polarity"]
        
        # Both texts should be somewhat subjective
        assert pos_features["subjectivity"] > 0
        assert neg_features["subjectivity"] > 0
    
    def test_extract_all_features(self):
        """Test complete feature extraction."""
        all_features = self.engineer.extract_all_features(self.test_text)
        
        # Check all feature categories are included
        assert "text_length" in all_features
        assert "word_count" in all_features
        assert "polarity" in all_features
        assert "subjectivity" in all_features
        assert "noun_count" in all_features
        assert "verb_count" in all_features
        
        # Check feature values are reasonable
        assert all_features["text_length"] > 0
        assert all_features["word_count"] > 0
        assert -1 <= all_features["polarity"] <= 1
        assert 0 <= all_features["subjectivity"] <= 1

class TestNewsDataPreprocessor:
    """Test cases for news data preprocessing."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'headline': ['Breaking News About Tech', 'Sports Update 2023', 'Politics Report'],
            'short_description': ['A new tech innovation has emerged.', 'Team wins championship!', 'New policy announced.'],
            'category': ['TECHNOLOGY', 'SPORTS', 'POLITICS']
        })
    
    def test_basic_cleaning(self, sample_df):
        """Test basic cleaning of news data."""
        preprocessor = NewsDataPreprocessor()
        result = preprocessor.basic_cleaning(sample_df)
        
        # Check text combination
        assert 'text' in result.columns
        assert result['text'].iloc[0] == 'Breaking News About Tech A new tech innovation has emerged.'
        
        # Check cleaning
        assert 'cleaned_text' in result.columns
        assert result['cleaned_text'].iloc[0] == result['cleaned_text'].iloc[0].lower()
        
        # Check no rows were incorrectly dropped
        assert len(result) == len(sample_df)
    
    def test_advanced_processing(self, sample_df):
        """Test advanced processing of news data."""
        preprocessor = NewsDataPreprocessor()
        df_basic = preprocessor.basic_cleaning(sample_df)
        result = preprocessor.advanced_processing(df_basic)
        
        # Check processed text exists
        assert 'processed_text' in result.columns
        
        # Check stopwords were removed
        for text in result['processed_text']:
            words = text.split()
            assert 'a' not in words
            assert 'the' not in words
    
    def test_feature_engineering(self, sample_df):
        """Test feature engineering for news data."""
        preprocessor = NewsDataPreprocessor()
        df_basic = preprocessor.basic_cleaning(sample_df)
        df_processed = preprocessor.advanced_processing(df_basic)
        result = preprocessor.feature_engineering(df_processed)
        
        # Check feature# filepath: c:\Users\pc\Desktop\nlp-project\story2audio\mlopsProjectnew\nlp-pipeline-project\tests\test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
import re
from pathlib import Path

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.text_cleaning import TextCleaner
from src.features.feature_engineering import FeatureEngineer
from src.data.preprocess import NewsDataPreprocessor

class TestTextCleaning:
    """Test cases for text cleaning functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
        self.test_text = "This is an Example text with Punctuation!!! 123 and URL: http://example.com"
    
    def test_basic_clean(self):
        """Test basic text cleaning."""
        cleaned = self.cleaner.basic_clean(self.test_text)
        
        # Check lowercase
        assert cleaned == cleaned.lower()
        
        # Check punctuation removed
        assert "!" not in cleaned
        assert ":" not in cleaned
        
        # Check numbers removed
        assert "123" not in cleaned
        
        # Check URL removed
        assert "http" not in cleaned
        assert "example.com" not in cleaned
        
        # Check content still exists
        assert "example text" in cleaned
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        text = "This is a test with some stopwords like the and a"
        cleaned = self.cleaner.remove_stopwords_from_text(text)
        
        # Check stopwords removed
        assert "this" not in cleaned.lower().split()
        assert "is" not in cleaned.lower().split()
        assert "a" not in cleaned.lower().split()
        assert "the" not in cleaned.lower().split()
        
        # Check content words still exist
        assert "test" in cleaned.lower().split()
        assert "stopwords" in cleaned.lower().split()
    
    def test_lemmatize_text(self):
        """Test lemmatization."""
        text = "The cats are running quickly through forests"
        lemmatized = self.cleaner.lemmatize_text(text)
        
        # Check lemmatization
        assert "cat" in lemmatized.split()
        assert "running" not in lemmatized.split()
        assert "run" in lemmatized.split()
        assert "forest" in lemmatized.split()
        assert "forests" not in lemmatized.split()
    
    def test_clean_text_full_pipeline(self):
        """Test full text cleaning pipeline."""
        original = "The CATS are running QUICKLY!!! through http://example.com forests. 123"
        processed = self.cleaner.clean_text(original)
        
        # Check results of full pipeline
        assert "cat" in processed.split()
        assert "run" in processed.split()
        assert "forest" in processed.split()
        assert "the" not in processed.split()
        assert "quickly" not in processed.split()  # Should be removed as stopword
        assert "http" not in processed
        assert "123" not in processed

class TestFeatureEngineering:
    """Test cases for feature engineering functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        self.test_text = "This is a sample text for feature extraction. It contains some sentiment words like happy and great!"
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        features = self.engineer.extract_basic_features(self.test_text)
        
        # Check basic features
        assert isinstance(features, dict)
        assert "text_length" in features
        assert features["text_length"] > 0
        assert features["text_length"] == len(self.test_text)
        assert features["word_count"] > 0
        assert features["avg_word_length"] > 0
    
    def test_extract_sentiment_features(self):
        """Test sentiment feature extraction."""
        # Positive text
        pos_text = "I am very happy and satisfied with the excellent results!"
        pos_features = self.engineer.extract_sentiment_features(pos_text)
        
        # Negative text
        neg_text = "I am disappointed and frustrated with the terrible outcome."
        neg_features = self.engineer.extract_sentiment_features(neg_text)
        
        # Check sentiment features
        assert isinstance(pos_features, dict)
        assert "polarity" in pos_features
        assert "subjectivity" in pos_features
        
        # Positive text should have higher polarity than negative
        assert pos_features["polarity"] > neg_features["polarity"]
        
        # Both texts should be somewhat subjective
        assert pos_features["subjectivity"] > 0
        assert neg_features["subjectivity"] > 0
    
    def test_extract_all_features(self):
        """Test complete feature extraction."""
        all_features = self.engineer.extract_all_features(self.test_text)
        
        # Check all feature categories are included
        assert "text_length" in all_features
        assert "word_count" in all_features
        assert "polarity" in all_features
        assert "subjectivity" in all_features
        assert "noun_count" in all_features
        assert "verb_count" in all_features
        
        # Check feature values are reasonable
        assert all_features["text_length"] > 0
        assert all_features["word_count"] > 0
        assert -1 <= all_features["polarity"] <= 1
        assert 0 <= all_features["subjectivity"] <= 1

class TestNewsDataPreprocessor:
    """Test cases for news data preprocessing."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'headline': ['Breaking News About Tech', 'Sports Update 2023', 'Politics Report'],
            'short_description': ['A new tech innovation has emerged.', 'Team wins championship!', 'New policy announced.'],
            'category': ['TECHNOLOGY', 'SPORTS', 'POLITICS']
        })
    
    def test_basic_cleaning(self, sample_df):
        """Test basic cleaning of news data."""
        preprocessor = NewsDataPreprocessor()
        result = preprocessor.basic_cleaning(sample_df)
        
        # Check text combination
        assert 'text' in result.columns
        assert result['text'].iloc[0] == 'Breaking News About Tech A new tech innovation has emerged.'
        
        # Check cleaning
        assert 'cleaned_text' in result.columns
        assert result['cleaned_text'].iloc[0] == result['cleaned_text'].iloc[0].lower()
        
        # Check no rows were incorrectly dropped
        assert len(result) == len(sample_df)
    
    def test_advanced_processing(self, sample_df):
        """Test advanced processing of news data."""
        preprocessor = NewsDataPreprocessor()
        df_basic = preprocessor.basic_cleaning(sample_df)
        result = preprocessor.advanced_processing(df_basic)
        
        # Check processed text exists
        assert 'processed_text' in result.columns
        
        # Check stopwords were removed
        for text in result['processed_text']:
            words = text.split()
            assert 'a' not in words
            assert 'the' not in words
    
    def test_feature_engineering(self, sample_df):
        """Test feature engineering for news data."""
        preprocessor = NewsDataPreprocessor()
        df_basic = preprocessor.basic_cleaning(sample_df)
        df_processed = preprocessor.advanced_processing(df_basic)
        result = preprocessor.feature_engineering(df_processed)
        
        # Check feature