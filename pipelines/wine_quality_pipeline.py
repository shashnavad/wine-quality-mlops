import kfp
from kfp import dsl
from kfp.dsl import component, InputPath, OutputPath
from typing import NamedTuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import json
import datetime

# Define base image for all components
BASE_IMAGE = 'shashnavad/wine-quality-mlops:latest'  # Updated to use Docker Hub

@component(base_image=BASE_IMAGE)
def download_data(dataset_path: OutputPath('CSV')) -> None:
    """Download wine quality dataset and perform initial validation."""
    import requests
    
    # Download dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    response = requests.get(url)
    
    # Save dataset
    with open(dataset_path, "wb") as f:
        f.write(response.content)
    
    print(f"Dataset downloaded to {dataset_path}")


@component(base_image=BASE_IMAGE)
def validate_data(
    dataset_path: InputPath('CSV'),
    validation_status_path: OutputPath('Text'),
    validation_report_path: OutputPath('JSON')
) -> None:
    """Validate data quality using basic checks."""
    import pandas as pd
    import json
    
    # Load data
    df = pd.read_csv(dataset_path, sep=";")
    
    # Create validation report
    validation_report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "summary_statistics": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std())
            } for col in df.columns
        }
    }
    
    # Check for validation issues
    validation_issues = []
    
    if df.isnull().any().any():
        validation_issues.append("Dataset contains missing values")
    
    if df["quality"].min() < 0 or df["quality"].max() > 10:
        validation_issues.append("Quality values outside expected range (0-10)")
    
    # Determine validation status
    validation_status = "passed" if not validation_issues else "failed: " + "; ".join(validation_issues)
    
    # Save outputs
    with open(validation_status_path, "w") as f:
        f.write(validation_status)
    
    with open(validation_report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"Validation status: {validation_status}")


@component(base_image=BASE_IMAGE)
def preprocess_data(
    dataset_path: InputPath('CSV'),
    features_path: OutputPath('NPY'),
    labels_path: OutputPath('NPY'),
    scaler_path: OutputPath('Pickle')
) -> None:
    """Preprocess the wine quality dataset."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Load data
    df = pd.read_csv(dataset_path, sep=";")
    
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


@component(base_image=BASE_IMAGE)
def train_model(
    features_path: InputPath('NPY'),
    labels_path: InputPath('NPY'),
    hyperparameters: dict,
    model_path: OutputPath('Pickle')
) -> None:
    """Train a machine learning model."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import joblib
    
    # Load data
    X = np.load(features_path)
    y = np.load(labels_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(**hyperparameters)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


@component(base_image=BASE_IMAGE)
def evaluate_model(
    model_path: InputPath('Pickle'),
    features_path: InputPath('NPY'),
    labels_path: InputPath('NPY'),
    metrics_path: OutputPath('JSON')
) -> None:
    """Evaluate the trained model."""
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    import json
    
    # Load data and model
    X = np.load(features_path)
    y = np.load(labels_path)
    model = joblib.load(model_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model evaluation metrics: {metrics}")


@component(base_image=BASE_IMAGE)
def deploy_model(
    model_path: InputPath('Pickle'),
    scaler_path: InputPath('Pickle'),
    metrics_path: InputPath('JSON'),
    deployment_path: OutputPath('Directory')
) -> None:
    """Package the model for deployment."""
    import shutil
    import os
    import json
    
    # Create deployment directory structure
    os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
    
    # Copy model and scaler
    shutil.copy(model_path, os.path.join(deployment_path, "model.joblib"))
    shutil.copy(scaler_path, os.path.join(deployment_path, "scaler.joblib"))
    
    # Copy metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(os.path.join(deployment_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create deployment info
    deployment_info = {
        "model_file": "model.joblib",
        "scaler_file": "scaler.joblib",
        "metrics_file": "metrics.json",
        "deployment_time": datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(deployment_path, "deployment_info.json"), 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"Model deployment package created at {deployment_path}")


@dsl.pipeline(
    name='Wine Quality Pipeline',
    description='End-to-end ML pipeline for wine quality prediction'
)
def wine_quality_pipeline(
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
    
    # Download data
    download_task = download_data()
    
    # Validate data
    validate_task = validate_data(
        dataset_path=download_task.outputs['dataset_path']
    )
    
    # Preprocess data
    preprocess_task = preprocess_data(
        dataset_path=download_task.outputs['dataset_path']
    )
    
    # Train model
    train_task = train_model(
        features_path=preprocess_task.outputs['features_path'],
        labels_path=preprocess_task.outputs['labels_path'],
        hyperparameters=hyperparameters
    )
    
    # Evaluate model
    evaluate_task = evaluate_model(
        model_path=train_task.outputs['model_path'],
        features_path=preprocess_task.outputs['features_path'],
        labels_path=preprocess_task.outputs['labels_path']
    )
    
    # Deploy model
    deploy_task = deploy_model(
        model_path=train_task.outputs['model_path'],
        scaler_path=preprocess_task.outputs['scaler_path'],
        metrics_path=evaluate_task.outputs['metrics_path']
    )


if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(wine_quality_pipeline, 'wine_quality_pipeline.yaml') 