#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import mlflow
import os

def train_model(features_path: str, labels_path: str, model_path: str, metrics_path: str, hyperparameters: str):
    """Train a machine learning model."""
    # Load data
    X = np.load(features_path)
    y = np.load(labels_path)
    
    # Parse hyperparameters
    params = json.loads(hyperparameters)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set MLflow tracking URI from environment variable
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-service:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("train_r2", train_score)
        mlflow.log_metric("test_r2", test_score)
        
        # Save metrics
        metrics = {
            'train_r2': float(train_score),
            'test_r2': float(test_score)
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
        print(f"Training metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wine quality model")
    parser.add_argument("--features_path", type=str, required=True, help="Path to processed features")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to processed labels")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to save metrics")
    parser.add_argument("--hyperparameters", type=str, required=True, help="JSON string of hyperparameters")
    
    args = parser.parse_args()
    
    train_model(
        features_path=args.features_path,
        labels_path=args.labels_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        hyperparameters=args.hyperparameters
    ) 