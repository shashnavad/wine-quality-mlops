import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import mlflow
import os
from kfp import dsl

@dsl.component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest',
    packages_to_install=['numpy', 'scikit-learn', 'joblib', 'mlflow']
)
def train(features_path: str, labels_path: str, hyperparameters: dict) -> dict:
    """Train a RandomForestRegressor model."""
    # Create output directories
    os.makedirs('/tmp/models', exist_ok=True)
    os.makedirs('/tmp/metrics', exist_ok=True)
    
    # Define output paths
    model_path = '/tmp/models/model.joblib'
    metrics_path = '/tmp/metrics/metrics.json'
    
    # Load data
    X = np.load(features_path)
    y = np.load(labels_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(**hyperparameters)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Save metrics
    metrics = {
        'train_r2': float(train_score),
        'test_r2': float(test_score)
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Log to MLflow if tracking URI is set
    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        with mlflow.start_run():
            mlflow.log_params(hyperparameters)
            mlflow.log_metric("train_r2", train_score)
            mlflow.log_metric("test_r2", test_score)
            mlflow.sklearn.log_model(model, "model")
    
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Training metrics: {metrics}")
    
    return {
        'model': model_path,
        'metrics': metrics_path
    } 