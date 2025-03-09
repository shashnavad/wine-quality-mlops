from kfp import dsl
from kfp import compiler
from components.preprocess import preprocess
from components.train import train

@dsl.pipeline(
    name='Wine Quality Pipeline',
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
    
    # Create output paths
    features_path = '/tmp/processed/features.npy'
    labels_path = '/tmp/processed/labels.npy'
    scaler_path = '/tmp/processed/scaler.joblib'
    model_path = '/tmp/models/model.joblib'
    metrics_path = '/tmp/metrics/metrics.json'
    
    # Preprocess data
    preprocess_task = preprocess(data_path=data_path)
    
    # Train and evaluate model
    train_task = train(
        features_path=features_path,
        labels_path=labels_path,
        hyperparameters=hyperparameters
    ).after(preprocess_task)
    
    # Set MLflow environment variable
    train_task.set_env_variable('MLFLOW_TRACKING_URI', 'http://mlflow-service:5000')

if __name__ == '__main__':
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    ) 