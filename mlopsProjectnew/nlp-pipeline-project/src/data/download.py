import os
import logging
from pathlib import Path
import json
import pandas as pd
import requests
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_NAME = "rmisra/news-category-dataset"
DATASET_FILE = "news_category_dataset_v3.json"

def download_dataset(output_dir=None, use_kaggle_api=True):
    """
    Download the News Category Dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset (default: project's raw data directory)
        use_kaggle_api: Whether to use Kaggle API or direct download
        
    Returns:
        Path to the downloaded dataset file
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = Path(output_dir) / "news_category_dataset.json"
    
    # Check if file already exists
    if output_file.exists():
        logger.info(f"Dataset already exists at {output_file}")
        return output_file
    
    try:
        if use_kaggle_api:
            logger.info(f"Downloading dataset '{DATASET_NAME}' using Kaggle API")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(DATASET_NAME, path=output_dir, unzip=True)
            
            # Rename the file if necessary
            downloaded_file = Path(output_dir) / DATASET_FILE
            if downloaded_file.exists() and downloaded_file != output_file:
                downloaded_file.rename(output_file)
                
        else:
            # Alternative: Direct download from a URL (if available)
            # This is a fallback option as Kaggle typically requires authentication
            logger.warning("Kaggle API disabled, using direct download (may not work)")
            url = "https://storage.googleapis.com/kaggle-data-sets/13/17/compressed/news_category_dataset_v2.json.zip"
            
            zip_path = Path(output_dir) / "news_category_dataset.zip"
            
            # Download the zip file
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove the zip file
            os.remove(zip_path)
        
        logger.info(f"Dataset downloaded successfully to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def load_and_preview_dataset(file_path=None):
    """
    Load and preview the dataset.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        DataFrame with the dataset
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "news_category_dataset.json"
    
    try:
        logger.info(f"Loading dataset from {file_path}")
        
        # Read JSON file line by line
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        df = pd.DataFrame(data)
        
        # Print dataset info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Categories: {df['category'].nunique()}")
        
        # Preview categories
        logger.info("\nCategory distribution:")
        category_counts = df['category'].value_counts().head(10)
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Download dataset
    file_path = download_dataset()
    
    # Load and preview dataset
    df = load_and_preview_dataset(file_path)