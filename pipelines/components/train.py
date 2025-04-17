import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import mlflow
import os
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output


@dsl.component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def train(
    features: Input[Dataset],
    labels: Input[Dataset],
    hyperparameters: dict,
    model: Output[Model],
    metrics: Output[Metrics],
    scaler: Input[Model]
):
    """Train a RandomForestRegressor model."""
    import os
    import joblib
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import mlflow
    
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

    # Only log to MLflow if not in testing mode and if MLflow server is available
    if not os.environ.get("TESTING", "False").lower() == "true":
        try:
            # Set a timeout for MLflow connection attempts
            import socket
            socket.setdefaulttimeout(5)  # 5 second timeout
            
            # Try to connect to MLflow server
            mlflow.set_tracking_uri('http://mlflow-service.mlops.svc.cluster.local:5000')
            
            with mlflow.start_run():
                mlflow.log_params(hyperparameters)
                mlflow.log_metric("train_r2", train_score)
                mlflow.log_metric("test_r2", test_score)
                mlflow.sklearn.log_model(model_obj, "model")
                mlflow.log_artifact(scaler.path, "preprocessor")
            
            print("Successfully logged metrics to MLflow")
        except Exception as e:
            print(f"MLflow logging failed (this is expected in local environments): {e}")
            print("Continuing without MLflow logging")
    else:
        print("Testing mode active, skipping MLflow logging")

    print(f"Model saved to {model.path}")
    print(f"Metrics saved to {metrics.path}")
    print(f"Training metrics: {metrics_dict}")