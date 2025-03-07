import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from typing import NamedTuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import os


@create_component_from_func
def download_data() -> NamedTuple('Outputs', [('dataset_path', str)]):
    """Download wine quality dataset and perform initial validation."""
    import pandas as pd
    import requests
    import os
    from collections import namedtuple
    
    # Create data directory
    os.makedirs("/tmp/data", exist_ok=True)
    
    # Download dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    response = requests.get(url)
    
    # Save dataset
    dataset_path = "/tmp/data/winequality-red.csv"
    with open(dataset_path, "wb") as f:
        f.write(response.content)
    
    print(f"Dataset downloaded to {dataset_path}")
    
    Outputs = namedtuple('Outputs', ['dataset_path'])
    return Outputs(dataset_path)


@create_component_from_func
def validate_data(dataset_path: str) -> NamedTuple('Outputs', [('validation_status', str), ('validation_report', str)]):
    """Validate data quality using basic checks."""
    import pandas as pd
    import json
    import os
    from collections import namedtuple
    
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
    
    # Check for missing values
    if df.isnull().any().any():
        validation_issues.append("Dataset contains missing values")
    
    # Check for quality range
    if df["quality"].min() < 0 or df["quality"].max() > 10:
        validation_issues.append("Quality values outside expected range (0-10)")
    
    # Determine validation status
    if validation_issues:
        validation_status = "failed: " + "; ".join(validation_issues)
    else:
        validation_status = "passed"
    
    # Save validation report
    os.makedirs("/tmp/reports", exist_ok=True)
    report_path = "/tmp/reports/validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"Validation status: {validation_status}")
    print(f"Validation report saved to {report_path}")
    
    Outputs = namedtuple('Outputs', ['validation_status', 'validation_report'])
    return Outputs(validation_status, json.dumps(validation_report))


@create_component_from_func
def preprocess_data(dataset_path: str) -> NamedTuple('Outputs', [('features_path', str), ('labels_path', str), ('scaler_path', str)]):
    """Preprocess the wine quality dataset."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    import os
    from collections import namedtuple
    
    # Load data
    df = pd.read_csv(dataset_path, sep=";")
    
    # Split features and labels
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create output directory
    os.makedirs("/tmp/processed", exist_ok=True)
    
    # Save processed data
    features_path = "/tmp/processed/features.npy"
    labels_path = "/tmp/processed/labels.npy"
    scaler_path = "/tmp/processed/scaler.joblib"
    
    np.save(features_path, X_scaled)
    np.save(labels_path, y.values)
    joblib.dump(scaler, scaler_path)
    
    print(f"Preprocessed features saved to {features_path}")
    print(f"Preprocessed labels saved to {labels_path}")
    print(f"Scaler saved to {scaler_path}")
    
    Outputs = namedtuple('Outputs', ['features_path', 'labels_path', 'scaler_path'])
    return Outputs(features_path, labels_path, scaler_path)


@create_component_from_func
def train_model(features_path: str, labels_path: str, hyperparameters: str) -> NamedTuple('Outputs', [('model_path', str)]):
    """Train a machine learning model."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import joblib
    import json
    import os
    from collections import namedtuple
    
    # Load data
    X = np.load(features_path)
    y = np.load(labels_path)
    
    # Parse hyperparameters
    params = json.loads(hyperparameters)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Create output directory
    os.makedirs("/tmp/models", exist_ok=True)
    
    # Save model
    model_path = "/tmp/models/model.joblib"
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")
    
    Outputs = namedtuple('Outputs', ['model_path'])
    return Outputs(model_path)


@create_component_from_func
def evaluate_model(model_path: str, features_path: str, labels_path: str) -> NamedTuple('Outputs', [('metrics_path', str)]):
    """Evaluate the trained model."""
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import json
    import os
    from collections import namedtuple
    
    # Load data and model
    X = np.load(features_path)
    y = np.load(labels_path)
    model = joblib.load(model_path)
    
    # Split data
    from sklearn.model_selection import train_test_split
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
    
    # Create output directory
    os.makedirs("/tmp/metrics", exist_ok=True)
    
    # Save metrics
    metrics_path = "/tmp/metrics/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"Model evaluation metrics: {metrics}")
    
    Outputs = namedtuple('Outputs', ['metrics_path'])
    return Outputs(metrics_path)


@create_component_from_func
def deploy_model(model_path: str, scaler_path: str, metrics_path: str) -> NamedTuple('Outputs', [('deployment_path', str)]):
    """Package the model for deployment."""
    import joblib
    import json
    import os
    import shutil
    from collections import namedtuple
    
    # Create deployment directory
    deployment_dir = "/tmp/deployment"
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Copy model and scaler
    shutil.copy(model_path, os.path.join(deployment_dir, "model.joblib"))
    shutil.copy(scaler_path, os.path.join(deployment_dir, "scaler.joblib"))
    
    # Load and save metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(os.path.join(deployment_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create deployment info
    deployment_info = {
        "model_file": "model.joblib",
        "scaler_file": "scaler.joblib",
        "metrics_file": "metrics.json",
        "api_endpoint": "/predict",
        "input_format": {
            "fixed_acidity": "float",
            "volatile_acidity": "float",
            "citric_acid": "float",
            "residual_sugar": "float",
            "chlorides": "float",
            "free_sulfur_dioxide": "float",
            "total_sulfur_dioxide": "float",
            "density": "float",
            "pH": "float",
            "sulphates": "float",
            "alcohol": "float"
        },
        "output_format": {
            "quality": "float",
            "confidence": "float"
        }
    }
    
    # Save deployment info
    with open(os.path.join(deployment_dir, "deployment_info.json"), 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"Model packaged for deployment in {deployment_dir}")
    
    Outputs = namedtuple('Outputs', ['deployment_path'])
    return Outputs(deployment_dir)


@dsl.pipeline(
    name='Wine Quality Pipeline',
    description='End-to-end ML pipeline for wine quality prediction'
)
def wine_quality_pipeline(
    n_estimators: int = 100,
    max_depth: str = "None",
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
):
    """Define the wine quality ML pipeline."""
    
    # Convert max_depth to proper format for JSON
    max_depth_value = None if max_depth == "None" else int(max_depth)
    
    # Create hyperparameters JSON
    hyperparameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth_value,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state
    }
    
    # Download and validate data
    download_task = download_data()
    validate_task = validate_data(download_task.outputs["dataset_path"])
    
    # Add validation check
    with dsl.Condition(validate_task.outputs["validation_status"] == "passed"):
        # Preprocess data
        preprocess_task = preprocess_data(download_task.outputs["dataset_path"])
        
        # Train model
        train_task = train_model(
            preprocess_task.outputs["features_path"],
            preprocess_task.outputs["labels_path"],
            hyperparameters=str(hyperparameters).replace("'", "\"").replace("None", "null")
        )
        
        # Evaluate model
        evaluate_task = evaluate_model(
            train_task.outputs["model_path"],
            preprocess_task.outputs["features_path"],
            preprocess_task.outputs["labels_path"]
        )
        
        # Deploy model
        deploy_task = deploy_model(
            train_task.outputs["model_path"],
            preprocess_task.outputs["scaler_path"],
            evaluate_task.outputs["metrics_path"]
        )


if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(wine_quality_pipeline, 'wine_quality_pipeline.yaml') 