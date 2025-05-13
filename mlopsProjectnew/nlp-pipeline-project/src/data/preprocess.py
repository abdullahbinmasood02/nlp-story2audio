import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

class NewsDataPreprocessor:
    """Class for preprocessing the News Category Dataset."""
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename="news_category_dataset.csv"):
        """Load the dataset."""
        file_path = RAW_DATA_DIR / filename
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def basic_cleaning(self, df):
        """
        Perform basic text cleaning:
        - Convert to lowercase
        - Remove punctuation
        - Remove extra whitespace
        """
        logger.info("Performing basic cleaning...")
        
        # Combine headline and short description
        df['text'] = df['headline'] + " " + df['short_description']
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self._basic_clean_text)
        
        # Remove any rows with empty text after cleaning
        original_count = len(df)
        df = df.dropna(subset=['cleaned_text'])
        df = df[df['cleaned_text'].str.strip() != ""]
        logger.info(f"Removed {original_count - len(df)} rows with empty text")
        
        return df
    
    def _basic_clean_text(self, text):
        """Basic text cleaning helper function."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_processing(self, df):
        """
        Perform advanced text processing:
        - Remove stopwords
        - Lemmatization
        """
        logger.info("Performing advanced text processing...")
        
        df['processed_text'] = df['cleaned_text'].apply(self._advanced_process_text)
        
        return df
    
    def _advanced_process_text(self, text):
        """Advanced text processing helper function."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stopwords]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def feature_engineering(self, df):
        """
        Extract additional features from text:
        - Text length
        - Word count
        - Average word length
        - Sentiment scores
        """
        logger.info("Performing feature engineering...")
        
        # Text length
        df['text_length'] = df['cleaned_text'].apply(len)
        
        # Word count
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        
        # Average word length
        df['avg_word_length'] = df['cleaned_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Sentiment analysis using TextBlob
        sentiment = df['cleaned_text'].apply(self._get_sentiment)
        df['sentiment_polarity'] = [s[0] for s in sentiment]
        df['sentiment_subjectivity'] = [s[1] for s in sentiment]
        
        return df
    
    def _get_sentiment(self, text):
        """Calculate sentiment scores using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0, 0
    
    def save_processed_data(self, df, filename="processed_news_data.csv"):
        """Save the processed dataset."""
        try:
            output_path = PROCESSED_DATA_DIR / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            # Also save a sample for quick testing
            sample_path = PROCESSED_DATA_DIR / "sample_processed_news_data.csv"
            df.sample(min(1000, len(df))).to_csv(sample_path, index=False)
            logger.info(f"Saved sample processed data to {sample_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def process(self):
        """Run the full preprocessing pipeline."""
        try:
            df = self.load_data()
            df = self.basic_cleaning(df)
            df = self.advanced_processing(df)
            df = self.feature_engineering(df)
            self.save_processed_data(df)
            
            return df
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    preprocessor = NewsDataPreprocessor()
    preprocessor.process()