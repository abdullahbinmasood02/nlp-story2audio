import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata

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

class TextCleaner:
    """A class for text cleaning operations."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """
        Initialize the TextCleaner.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to perform lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def basic_clean(self, text):
        """
        Perform basic text cleaning.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords_from_text(self, text):
        """
        Remove English stopwords from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with stopwords removed
        """
        if not text:
            return ""
            
        tokens = text.split()
        tokens = [token for token in tokens if token.lower() not in self.stopwords]
        return " ".join(tokens)
    
    def lemmatize_text(self, text):
        """
        Lemmatize text.
        
        Args:
            text: Input text string
            
        Returns:
            Lemmatized text
        """
        if not text:
            return ""
            
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)
    
    def clean_text(self, text):
        """
        Apply full text cleaning pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            Fully cleaned and processed text
        """
        # Basic cleaning
        text = self.basic_clean(text)
        
        # Stopword removal (optional)
        if self.remove_stopwords:
            text = self.remove_stopwords_from_text(text)
        
        # Lemmatization (optional)
        if self.lemmatize:
            text = self.lemmatize_text(text)
        
        return text

    def clean_text_batch(self, texts):
        """
        Clean a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]