import pandas as pd
import numpy as np

def create_drifted_data():
    """Create synthetic drifted data to test validation."""
    # Load original data
    original_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
    
    # Introduce drift
    drifted_df = original_df.copy()
    drifted_df['alcohol'] = drifted_df['alcohol'] * 1.3  # 30% increase in alcohol content
    drifted_df['pH'] = drifted_df['pH'] + 0.5  # Shift pH values
    
    # Save drifted data
    drifted_df.to_csv("data/drifted_wine_data.csv", sep=";", index=False)
    print("Created drifted data at: data/drifted_wine_data.csv")
    
    return drifted_df

if __name__ == "__main__":
    # Create directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Create drifted data
    create_drifted_data()
    
    print("\nTo test data drift detection, run:")
    print("python run_pipeline.py --data-path data/drifted_wine_data.csv")