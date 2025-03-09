#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(data_path: str, features_path: str, labels_path: str, scaler_path: str):
    """Preprocess the wine quality dataset."""
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess wine quality data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--features_path", type=str, required=True, help="Path to save processed features")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to save processed labels")
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to save fitted scaler")
    
    args = parser.parse_args()
    
    preprocess_data(
        data_path=args.data_path,
        features_path=args.features_path,
        labels_path=args.labels_path,
        scaler_path=args.scaler_path
    ) 