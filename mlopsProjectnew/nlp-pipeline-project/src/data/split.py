import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RANDOM_SEED = 42

def split_dataset(input_file="processed_news_data.csv", test_size=0.2, valid_size=0.1):
    """
    Split the processed dataset into train, validation, and test sets.
    
    Args:
        input_file: Name of the processed data file
        test_size: Proportion of data to use for testing
        valid_size: Proportion of data to use for validation
        
    Returns:
        Tuple of paths to train, validation, and test files
    """
    try:
        # Create directories if they don't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        input_path = PROCESSED_DATA_DIR / input_file
        logger.info(f"Loading processed data from {input_path}")
        
        # Load processed data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Calculate validation size relative to training set
        train_valid_size = 1 - test_size
        relative_valid_size = valid_size / train_valid_size
        
        # First split: train+validation vs test
        train_valid_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=RANDOM_SEED,
            stratify=df['category']
        )
        
        # Second split: train vs validation
        train_df, valid_df = train_test_split(
            train_valid_df, 
            test_size=relative_valid_size, 
            random_state=RANDOM_SEED,
            stratify=train_valid_df['category']
        )
        
        # Save the splits
        train_path = PROCESSED_DATA_DIR / "train.csv"
        valid_path = PROCESSED_DATA_DIR / "validation.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Log split information
        logger.info(f"Data split complete:")
        logger.info(f"  Train set: {len(train_df)} rows, saved to {train_path}")
        logger.info(f"  Validation set: {len(valid_df)} rows, saved to {valid_path}")
        logger.info(f"  Test set: {len(test_df)} rows, saved to {test_path}")
        
        # Create metadata file
        metadata = {
            'original_size': len(df),
            'train_size': len(train_df),
            'validation_size': len(valid_df),
            'test_size': len(test_df),
            'random_seed': RANDOM_SEED,
            'test_size_param': test_size,
            'valid_size_param': valid_size,
            'feature_columns': [col for col in df.columns if col not in ['category']],
            'target_column': 'category',
            'categorical_columns': ['category'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = PROCESSED_DATA_DIR / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Split metadata saved to {metadata_path}")
        
        return train_path, valid_path, test_path
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise

if __name__ == "__main__":
    split_dataset()