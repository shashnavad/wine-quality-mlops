import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from kfp import dsl

@dsl.component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest',
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'joblib']
)
def preprocess(data_path: str) -> dict:
    """Preprocess the wine quality data."""
    # Create output directories
    os.makedirs('/tmp/processed', exist_ok=True)
    
    # Define output paths
    features_path = '/tmp/processed/features.npy'
    labels_path = '/tmp/processed/labels.npy'
    scaler_path = '/tmp/processed/scaler.joblib'
    
    # Load data
    df = pd.read_csv(data_path, sep=";")
    
    # Split features and labels
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save processed data
    np.save(features_path, X_scaled)
    np.save(labels_path, y.values)
    joblib.dump(scaler, scaler_path)
    
    print(f"Preprocessed features saved to {features_path}")
    print(f"Preprocessed labels saved to {labels_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return {
        'features': features_path,
        'labels': labels_path,
        'scaler': scaler_path
    } 