import os
import sys

# Set environment variables to disable Git functionality
os.environ["GE_USAGE_STATISTICS_ENABLED"] = "False"
os.environ["GE_HOME"] = "/tmp/great_expectations_home"
os.environ["GE_CONFIG_VERSION"] = "3"  # Force using V3 config
os.environ["GE_UNCOMMITTED_DIRECTORIES"] = "True"  # Skip Git checks
os.environ["GX_ASSUME_MISSING_LIBRARIES"] = "git"

# Mock out git before anything tries to import it
class MockGit:
    class Repo:
        @staticmethod
        def init(*args, **kwargs):
            pass
    
    class NoSuchPathError(Exception):
        pass

# Add the mock to sys.modules
sys.modules['git'] = MockGit

# Now continue with the regular imports
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
from typing import List
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
    base_image='pes1ug19cs601/wine-quality-mlops:latest',
    packages_to_install=['xgboost', 'lightgbm']
)
def train(
    features: Input[Dataset],
    labels: Input[Dataset],
    hyperparameters: dict,
    model: Output[Model],
    metrics: Output[Metrics],
    scaler: Input[Model],
    model_type: str = "RandomForest"  # Default argument moved to the end
):
    """Train a model with support for multiple algorithms."""
    import os
    import joblib
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import lightgbm as lgb
    import mlflow
    import time
    from prometheus_client import start_http_server, Gauge

    # Load data
    X = np.load(features.path)
    y = np.load(labels.path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model based on model_type
    print(f"Training {model_type} model with hyperparameters: {hyperparameters}")
    if model_type == "RandomForest":
        model_obj = RandomForestRegressor(**hyperparameters)
    elif model_type == "XGBoost":
        model_obj = xgb.XGBRegressor(**hyperparameters)
    elif model_type == "LightGBM":
        model_obj = lgb.LGBMRegressor(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    start_time = time.time()
    # Train model
    model_obj.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model_obj.score(X_train, y_train)
    test_score = model_obj.score(X_test, y_test)
    
    # Save metrics
    metrics_dict = {
        'model_type': model_type,
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
                mlflow.log_param("model_type", model_type)
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
    # Add Prometheus metrics server (after model evaluation)
    prom_port = int(os.getenv("PROMETHEUS_PORT", 8000))
    start_http_server(prom_port)
    
    # Create Prometheus gauges
    train_r2_gauge = Gauge('train_r2_score', 'Training R² score', ['model_type'])
    test_r2_gauge = Gauge('test_r2_score', 'Test R² score', ['model_type'])
    training_time_gauge = Gauge('training_time_seconds', 'Training duration')
    
    # Set metric values
    train_r2_gauge.labels(model_type=model_type).set(train_score)
    test_r2_gauge.labels(model_type=model_type).set(test_score)
    training_time_gauge.set(time.time() - start_time)

@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def select_best_model(
    best_model: Output[Model],  
    best_metrics: Output[Metrics],  
    model1: Input[Model] = None,  
    model2: Input[Model] = None, 
    model3: Input[Model] = None,  
    metrics1: Input[Metrics] = None,  
    metrics2: Input[Metrics] = None,  
    metrics3: Input[Metrics] = None  
):
    """Select the best model based on test R² score."""
    import json
    import shutil
    import copy
    
    best_score = -float('inf')
    best_model_idx = -1
    all_metrics_summary = []
    
    # Create lists of non-None inputs
    models = [m for m in [model1, model2, model3] if m is not None]
    metrics_files = [m for m in [metrics1, metrics2, metrics3] if m is not None]
    
    # Load all metrics and find the best model
    for i, metric_file in enumerate(metrics_files):
        with open(metric_file.path, 'r') as f:
            metric_data = json.load(f)
            
        # Create a simplified summary to avoid circular references
        metric_summary = {
            "model_type": metric_data.get("model_type", "Unknown"),
            "test_r2": float(metric_data.get("test_r2", -1.0)),
            "train_r2": float(metric_data.get("train_r2", -1.0))
        }
        
        all_metrics_summary.append(metric_summary)
        test_r2 = metric_summary["test_r2"]
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model_idx = i
    
    # Copy the best model and its metrics
    if best_model_idx >= 0:
        shutil.copy(models[best_model_idx].path, best_model.path)
        
        # Create a new dictionary for best metrics to avoid reference issues
        best_metric_data = copy.deepcopy(all_metrics_summary[best_model_idx])
        best_metric_data["model_comparison"] = [
            {"model": m["model_type"], "test_r2": m["test_r2"]}
            for m in all_metrics_summary
        ]
        
        with open(best_metrics.path, 'w') as f:
            json.dump(best_metric_data, f, indent=2)
            
        print(f"Selected best model: {best_metric_data['model_type']}")
        print(f"Best test R²: {best_score:.4f}")
    else:
        raise ValueError("No valid models found")


@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def validate_data(
    data_path: str,
    metrics: Output[Metrics],
    validation_success: Output[bool]
):
    """Validate wine data for drift using Great Expectations with Git functionality disabled."""
    import os
    import sys
    import json
    import pandas as pd

    import logging
    # Configure root logger and specific loggers to suppress debug messages
    logging.getLogger().setLevel(logging.ERROR)  # Set root logger to ERROR
    logging.getLogger('great_expectations').setLevel(logging.ERROR)
    logging.getLogger('great_expectations.expectations.registry').setLevel(logging.CRITICAL)  # Even stricter for registry

    
    # Set environment variables
    os.environ["GE_USAGE_STATISTICS_ENABLED"] = "False"
    os.environ["GE_UNCOMMITTED_DIRECTORIES"] = "True"
    os.environ["GX_ASSUME_MISSING_LIBRARIES"] = "git"
    
    
    # Import Great Expectations components
    from great_expectations.data_context.types.base import DataContextConfig
    from great_expectations.data_context import BaseDataContext
    from great_expectations.core.batch import RuntimeBatchRequest
    
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Create context configuration
        print("Creating context configuration...")
        context_config = DataContextConfig(
            store_backend_defaults=None,
            checkpoint_store_name=None,
            datasources={
                "pandas_datasource": {
                    "class_name": "Datasource",
                    "module_name": "great_expectations.datasource",
                    "execution_engine": {
                        "class_name": "PandasExecutionEngine",
                        "module_name": "great_expectations.execution_engine"
                    },
                    "data_connectors": {
                        "runtime_connector": {
                            "class_name": "RuntimeDataConnector",
                            "module_name": "great_expectations.datasource.data_connector",
                            "batch_identifiers": ["batch_id"]
                        }
                    }
                }
            }
        )
        
        # Create context
        print("Creating context...")
        context = BaseDataContext(project_config=context_config)
        
        # Create expectation suite
        print("Creating expectation suite...")
        suite_name = "wine_quality_suite"
        context.create_expectation_suite(suite_name, overwrite_existing=True)
        
        # Create batch request
        print("Creating batch request...")
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_connector",
            data_asset_name="wine_data",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"batch_id": "default_identifier"},
        )
        
        # Get validator
        print("Getting validator...")
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        # Add expectations
        print("Adding expectations...")
        expectations = []
        
        # Check that columns match expected list
        expectations.append(validator.expect_table_columns_to_match_ordered_list(list(df.columns)))
        
        # Check data types
        expectations.append(validator.expect_column_values_to_be_of_type("fixed acidity", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("volatile acidity", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("citric acid", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("residual sugar", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("chlorides", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("free sulfur dioxide", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("total sulfur dioxide", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("density", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("pH", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("sulphates", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("alcohol", "float64"))
        expectations.append(validator.expect_column_values_to_be_of_type("quality", "int64"))
        
        # Check value ranges
        expectations.append(validator.expect_column_values_to_be_between("fixed acidity", min_value=3.8, max_value=15.9))
        expectations.append(validator.expect_column_values_to_be_between("volatile acidity", min_value=0.08, max_value=1.58))
        expectations.append(validator.expect_column_values_to_be_between("citric acid", min_value=0, max_value=1.66))
        expectations.append(validator.expect_column_values_to_be_between("residual sugar", min_value=0.6, max_value=65.8))
        expectations.append(validator.expect_column_values_to_be_between("chlorides", min_value=0.009, max_value=0.611))
        expectations.append(validator.expect_column_values_to_be_between("free sulfur dioxide", min_value=1, max_value=289))
        expectations.append(validator.expect_column_values_to_be_between("total sulfur dioxide", min_value=6, max_value=440))
        expectations.append(validator.expect_column_values_to_be_between("density", min_value=0.98711, max_value=1.03898))
        expectations.append(validator.expect_column_values_to_be_between("pH", min_value=2.72, max_value=4.01))
        expectations.append(validator.expect_column_values_to_be_between("sulphates", min_value=0.22, max_value=2))
        expectations.append(validator.expect_column_values_to_be_between("alcohol", min_value=8, max_value=14.9))
        expectations.append(validator.expect_column_values_to_be_between("quality", min_value=3, max_value=9))
        
        # Check for missing values
        for column in df.columns:
            expectations.append(validator.expect_column_values_to_not_be_null(column))
        
        # Run validation
        print("Running validation...")
        validation_results = validator.validate()
        
        # Process results
        print("Processing results...")
        validation_passed = validation_results.success
        
        # Prepare metrics
        validation_metrics = {
            "validation_success": validation_passed,
            "evaluated_expectations": validation_results.statistics["evaluated_expectations"],
            "successful_expectations": validation_results.statistics["successful_expectations"],
            "unsuccessful_expectations": validation_results.statistics["unsuccessful_expectations"],
        }
        
        # Log metrics
        print("Logging metrics...")
        metrics.log_metric("validation_success", float(validation_passed))
        metrics.log_metric("evaluated_expectations", float(validation_results.statistics["evaluated_expectations"]))
        metrics.log_metric("successful_expectations", float(validation_results.statistics["successful_expectations"]))
        metrics.log_metric("unsuccessful_expectations", float(validation_results.statistics["unsuccessful_expectations"]))
        
        print(f"Validation {'passed' if validation_passed else 'failed'}")
        print(f"Metrics: {validation_metrics}")
        
        validation_success = validation_passed
        with open(validation_success.path, 'w') as f:
            f.write(str(validation_passed).lower())
        return validation_metrics
        
    except Exception as e:
        print(f"Error in validate_data: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Log failure in metrics
        metrics.log_metric("validation_success", 0.0)
        metrics.log_metric("error", 1.0)
        with open(validation_success.path, 'w') as f:
            f.write("False")
        raise

@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest',
    packages_to_install=['kserve==0.10.1']
)
def deploy_model(
    model: Input[Model],
    metrics: Input[Metrics],
    scaler: Input[Model],
    service_name: str,
    namespace: str = "kubeflow"
) -> str:
    import os
    import json
    import pickle
    import tempfile
    from kubernetes import client
    from kubernetes import config
    from kserve import KServeClient
    from kserve import constants
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1SKLearnSpec
    
    # Create a temporary directory to prepare the model
    model_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(model_dir, "model"), exist_ok=True)
    
    # Load and save the model and scaler
    with open(model.path, 'rb') as f:
        model_obj = pickle.load(f)
        
    with open(scaler.path, 'rb') as f:
        scaler_obj = pickle.load(f)
    
    # Save model and scaler to the temporary directory
    with open(os.path.join(model_dir, "model", "model.pkl"), 'wb') as f:
        pickle.dump(model_obj, f)
        
    with open(os.path.join(model_dir, "model", "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler_obj, f)
    
    # Load metrics to include in model metadata
    with open(metrics.path, 'r') as f:
        metrics_data = json.load(f)
    
    # Create a metadata file
    metadata = {
        "name": service_name,
        "version": "v1",
        "metrics": metrics_data
    }
    
    with open(os.path.join(model_dir, "model", "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    # Create a simple inference script
    inference_script = """
import os
import pickle
import json
import numpy as np

class WineQualityModel(object):
    def __init__(self):
        self.model = None
        self.scaler = None
        self.ready = False
        
    def load(self):
        model_dir = os.path.join(os.getcwd(), "model")
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        self.ready = True
        
    def predict(self, X, feature_names=None):
        if not self.ready:
            self.load()
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions.tolist()
"""
    
    with open(os.path.join(model_dir, "model", "WineQualityModel.py"), 'w') as f:
        f.write(inference_script)
    
    # Create a simple requirements.txt
    with open(os.path.join(model_dir, "model", "requirements.txt"), 'w') as f:
        f.write("scikit-learn==1.0.2\nnumpy==1.22.3\n")
    
    # Upload the model to a storage location (MinIO, S3, etc.)
    # Assuming you have a PVC for model storage
    model_uri = f"pvc://{service_name}-models"
    
    # In a real implementation, you would upload the model to your storage
    # For now, printing the path and assuming it's accessible
    print(f"Model prepared at: {model_dir}")
    print(f"Model would be deployed from: {model_uri}")
    
    try:
        # Initialize KServe client
        config.load_incluster_config()
        kserve_client = KServeClient()
        
        # Define the inference service
        isvc = V1beta1InferenceService(
            api_version=constants.KSERVE_V1BETA1,
            kind=constants.KSERVE_KIND,
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=namespace,
                annotations={"sidecar.istio.io/inject": "false"}
            ),
            spec=V1beta1InferenceServiceSpec(
                predictor=V1beta1PredictorSpec(
                    sklearn=V1beta1SKLearnSpec(
                        storage_uri=model_uri,
                        resources=client.V1ResourceRequirements(
                            requests={"cpu": "100m", "memory": "1Gi"},
                            limits={"cpu": "1", "memory": "2Gi"}
                        )
                    )
                )
            )
        )
        
        # Create the inference service
        kserve_client.create(isvc)
        print(f"Inference service '{service_name}' created in namespace '{namespace}'")
        
        # Get the URL of the deployed service
        service_url = f"http://{service_name}.{namespace}.svc.cluster.local/v1/models/{service_name}:predict"
        return service_url
        
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        return f"Deployment failed: {str(e)}"

if os.environ.get("TESTING", "False").lower() != "true":
    @dsl.pipeline(
    name='wine-quality-pipeline',
    description='End-to-end ML pipeline for wine quality prediction with model selection'
)
    def wine_quality_pipeline(
        data_path: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        # Model types to train
        use_random_forest: bool = True,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,  
        # RandomForest parameters
        rf_n_estimators: int = 100,
        rf_max_depth: int = None,
        rf_min_samples_split: int = 2,
        # XGBoost parameters
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        # LightGBM parameters
        lgbm_n_estimators: int = 100,
        lgbm_max_depth: int = -1,
        lgbm_learning_rate: float = 0.1,
        # Service parameters
        service_name: str = 'wine-quality-predictor'
    ):
        """Define the wine quality prediction pipeline with model selection."""
        
        # Construct hyperparameters dictionaries from pipeline parameters
        hyperparameters = {
            "RandomForest": {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'min_samples_split': rf_min_samples_split,
                'random_state': 42
            },
            "XGBoost": {
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'learning_rate': xgb_learning_rate,
                'random_state': 42
            },
            "LightGBM": {
                'n_estimators': lgbm_n_estimators,
                'max_depth': lgbm_max_depth,
                'learning_rate': lgbm_learning_rate,
                'random_state': 42
            }
        }

        # Validate data for drift
        validation_task = validate_data(data_path=data_path)
        
        with dsl.Condition(validation_task.outputs["validation_success"] == True):
            # Preprocess data (only if validation passes)
            preprocess_task = preprocess(data_path=data_path).after(validation_task)
            
            # Create separate tasks for each model type to avoid using ParallelFor
            rf_task = None
            xgb_task = None
            lgbm_task = None
            
            # Create model training tasks individually
            if use_random_forest:
                rf_task = train(
                    features=preprocess_task.outputs['features'],
                    labels=preprocess_task.outputs['labels'],
                    hyperparameters=hyperparameters["RandomForest"],
                    model_type="RandomForest",
                    scaler=preprocess_task.outputs['scaler']
                ).set_env_variable(name="PROMETHEUS_PORT", value="8000")
                
            if use_xgboost:
                xgb_task = train(
                    features=preprocess_task.outputs['features'],
                    labels=preprocess_task.outputs['labels'],
                    hyperparameters=hyperparameters["XGBoost"],
                    model_type="XGBoost",
                    scaler=preprocess_task.outputs['scaler']
                ).set_env_variable(name="PROMETHEUS_PORT", value="8000")
            
            if use_lightgbm:
                lgbm_task = train(
                    features=preprocess_task.outputs['features'],
                    labels=preprocess_task.outputs['labels'],
                    hyperparameters=hyperparameters["LightGBM"],
                    model_type="LightGBM",
                    scaler=preprocess_task.outputs['scaler']
                ).set_env_variable(name="PROMETHEUS_PORT", value="8000")
            
            # Collect models and metrics from all active tasks
            model_list = []
            metrics_list = []
            
            # Add outputs from each model training task if it exists
            for task in [rf_task, xgb_task, lgbm_task]:
                if task is not None:
                    model_list.append(task.outputs['model'])
                    metrics_list.append(task.outputs['metrics'])
            
            # Select the best model using individual inputs
            select_model_task = select_best_model(
            model1=rf_task.outputs['model'] if rf_task else None,
            model2=xgb_task.outputs['model'] if xgb_task else None,
            model3=lgbm_task.outputs['model'] if lgbm_task else None,
            metrics1=rf_task.outputs['metrics'] if rf_task else None,
            metrics2=xgb_task.outputs['metrics'] if xgb_task else None,
            metrics3=lgbm_task.outputs['metrics'] if lgbm_task else None
        )

            
            # Deploy only the best model
            deploy_task = deploy_model(
                model=select_model_task.outputs['best_model'],
                metrics=select_model_task.outputs['best_metrics'],
                scaler=preprocess_task.outputs['scaler'],
                service_name=service_name
            )

    if __name__ == '__main__':
        # Compile the pipeline
        compiler.Compiler().compile(
            pipeline_func=wine_quality_pipeline,
            package_path='wine_quality_pipeline.yaml'
        ) 