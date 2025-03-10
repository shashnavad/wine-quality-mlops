from kfp import dsl
from kfp import compiler
from kfp.dsl import Dataset, Input, Output, Model, Metrics, component
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import os

@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def preprocess(
    data_path: str,
    features: Output[Dataset],
    labels: Output[Dataset],
    scaler: Output[Model]
):
    """Preprocess the wine quality data."""
    # Load data
    df = pd.read_csv(data_path, sep=";")

    # Split features and labels
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Scale features
    scaler_obj = StandardScaler()
    X_scaled = scaler_obj.fit_transform(X)

    # Save processed data
    np.save(features.path, X_scaled)
    np.save(labels.path, y.values)
    joblib.dump(scaler_obj, scaler.path)

    print(f"Preprocessed features saved to {features.path}")
    print(f"Preprocessed labels saved to {labels.path}")
    print(f"Scaler saved to {scaler.path}")

@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def train(
    features: Input[Dataset],
    labels: Input[Dataset],
    hyperparameters: dict,
    model: Output[Model],
    metrics: Output[Metrics]
):
    """Train a RandomForestRegressor model."""
    # Load data
    X = np.load(features.path)
    y = np.load(labels.path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model_obj = RandomForestRegressor(**hyperparameters)
    model_obj.fit(X_train, y_train)

    # Evaluate model
    train_score = model_obj.score(X_train, y_train)
    test_score = model_obj.score(X_test, y_test)

    # Save metrics
    metrics_dict = {
        'train_r2': float(train_score),
        'test_r2': float(test_score)
    }
    with open(metrics.path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Save model
    joblib.dump(model_obj, model.path)

    # Log to MLflow
    mlflow.set_tracking_uri('http://mlflow-service.mlops.svc.cluster.local:5000')
    with mlflow.start_run():
        mlflow.log_params(hyperparameters)
        mlflow.log_metric("train_r2", train_score)
        mlflow.log_metric("test_r2", test_score)
        mlflow.sklearn.log_model(model_obj, "model")

    print(f"Model saved to {model.path}")
    print(f"Metrics saved to {metrics.path}")
    print(f"Training metrics: {metrics_dict}")

@dsl.pipeline(
    name='wine-quality-pipeline',
    description='End-to-end ML pipeline for wine quality prediction'
)
def wine_quality_pipeline(
    data_path: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
):
    """Define the wine quality prediction pipeline."""
    
    # Convert hyperparameters to dictionary
    hyperparameters = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state
    }
    
    # Preprocess data
    preprocess_task = preprocess(data_path=data_path)
    
    # Train and evaluate model
    train_task = train(
        features=preprocess_task.outputs['features'],
        labels=preprocess_task.outputs['labels'],
        hyperparameters=hyperparameters
    )

if __name__ == '__main__':
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    ) 