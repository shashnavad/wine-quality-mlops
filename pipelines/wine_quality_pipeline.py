import kfp
from kfp import dsl
from kfp.dsl import ContainerOp
import json
import k8s_client

# Define base image
BASE_IMAGE = 'pes1ug19cs601/wine-quality-mlops:latest'

def preprocess_op(data_path: str) -> ContainerOp:
    """Creates a preprocessing component."""
    return ContainerOp(
        name='preprocess-data',
        image=BASE_IMAGE,
        command=['python', '/app/src/components/preprocess.py'],
        arguments=[
            '--data_path', data_path,
            '--features_path', '/tmp/processed/features.npy',
            '--labels_path', '/tmp/processed/labels.npy',
            '--scaler_path', '/tmp/processed/scaler.joblib'
        ],
        file_outputs={
            'features': '/tmp/processed/features.npy',
            'labels': '/tmp/processed/labels.npy',
            'scaler': '/tmp/processed/scaler.joblib'
        }
    )

def train_op(features_path: str, labels_path: str, hyperparameters: dict) -> ContainerOp:
    """Creates a training component."""
    return ContainerOp(
        name='train-model',
        image=BASE_IMAGE,
        command=['python', '/app/src/components/train.py'],
        arguments=[
            '--features_path', features_path,
            '--labels_path', labels_path,
            '--model_path', '/tmp/models/model.joblib',
            '--metrics_path', '/tmp/metrics/metrics.json',
            '--hyperparameters', json.dumps(hyperparameters)
        ],
        file_outputs={
            'model': '/tmp/models/model.joblib',
            'metrics': '/tmp/metrics/metrics.json'
        }
    )

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
    
    # Preprocess data
    preprocess_task = preprocess_op(data_path=data_path)
    
    # Train and evaluate model
    train_task = train_op(
        features_path=preprocess_task.outputs['features'],
        labels_path=preprocess_task.outputs['labels'],
        hyperparameters=hyperparameters
    )
    
    # Add volume mount for MLflow
    train_task.add_env_variable(
        k8s_client.V1EnvVar(
            name='MLFLOW_TRACKING_URI',
            value='http://mlflow-service:5000'
        )
    )

if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    ) 