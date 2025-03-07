import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys

# Add the parent directory to the path to import from src.data
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.data_loader import WineDataLoader


class ModelTrainer:
    """Model trainer for the Wine Quality dataset."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the model trainer.
        
        Args:
            model_dir: Directory to store the trained models.
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.data_loader = WineDataLoader()
    
    def train_model(
        self,
        model_type: str = "random_forest",
        params: Optional[Dict[str, Any]] = None,
        experiment_name: str = "wine-quality",
        log_to_mlflow: bool = True,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a machine learning model.
        
        Args:
            model_type: Type of model to train.
            params: Parameters for the model.
            experiment_name: Name of the MLflow experiment.
            log_to_mlflow: Whether to log to MLflow.
        
        Returns:
            Tuple of (trained model, evaluation metrics).
        """
        # Get data
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_data()
        
        # Set default parameters if not provided
        if params is None:
            if model_type == "random_forest":
                params = {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                }
        
        # Initialize MLflow
        if log_to_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_params(params)
        
        # Train model
        if model_type == "random_forest":
            model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Log metrics to MLflow
        if log_to_mlflow:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # End the MLflow run
            mlflow.end_run()
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        return model, metrics
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a machine learning model.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        
        return metrics
    
    def load_model(self, model_path: str) -> Any:
        """Load a trained model.
        
        Args:
            model_path: Path to the trained model.
        
        Returns:
            Loaded model.
        """
        return joblib.load(model_path)


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    model, metrics = trainer.train_model(log_to_mlflow=False)
    
    print("Model evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}") 