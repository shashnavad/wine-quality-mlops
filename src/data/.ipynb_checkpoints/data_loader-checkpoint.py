import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WineDataLoader:
    """Data loader for the Wine Quality dataset."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory to store the data.
        """
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, "winequality-red.csv")
        os.makedirs(data_dir, exist_ok=True)
    
    def download_data(self) -> str:
        """Download the Wine Quality dataset.
        
        Returns:
            Path to the downloaded dataset.
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        
        if not os.path.exists(self.dataset_path):
            print(f"Downloading dataset to {self.dataset_path}...")
            response = requests.get(url)
            with open(self.dataset_path, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print(f"Dataset already exists at {self.dataset_path}.")
        
        return self.dataset_path
    
    def load_data(self) -> pd.DataFrame:
        """Load the Wine Quality dataset.
        
        Returns:
            DataFrame containing the dataset.
        """
        if not os.path.exists(self.dataset_path):
            self.download_data()
        
        return pd.read_csv(self.dataset_path, sep=";")
    
    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the Wine Quality dataset.
        
        Args:
            df: DataFrame containing the dataset. If None, load the dataset.
        
        Returns:
            Tuple of (features, labels).
        """
        if df is None:
            df = self.load_data()
        
        # Split features and labels
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def get_train_test_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train and test data.
        
        Args:
            test_size: Proportion of the dataset to include in the test split.
            random_state: Random state for reproducibility.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X, y = self.preprocess_data()
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # Example usage
    loader = WineDataLoader()
    X_train, X_test, y_train, y_test = loader.get_train_test_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}") 