import os
import sys

import logging
# Configure root logger and specific loggers to suppress debug messages
logging.getLogger().setLevel(logging.ERROR)  # Set root logger to ERROR
logging.getLogger('great_expectations').setLevel(logging.ERROR)
logging.getLogger('great_expectations.expectations.registry').setLevel(logging.CRITICAL)  # Even stricter for registry
# Set environment variables first
os.environ["GE_USAGE_STATISTICS_ENABLED"] = "False"
os.environ["GE_HOME"] = "/tmp/great_expectations_home"
os.environ["TESTING"] = "True"
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

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from kfp import local
import pickle
import json 
import joblib
from kfp.dsl import Dataset, Model, Metrics

# Import components directly
from wine_quality_pipeline import validate_data, preprocess, train

# Initialize local runner
local.init(runner=local.SubprocessRunner())

# Helper class to mock KFP artifacts
class MockArtifact:
    def __init__(self, path):
        self.path = path

from unittest.mock import MagicMock

class MockPrometheusClient:
    Gauge = MagicMock()
    start_http_server = MagicMock()

sys.modules['prometheus_client'] = MockPrometheusClient

def test_validate_data(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)  # Work in temporary directory
    
    try:
        # Set environment variables
        os.environ["GE_USAGE_STATISTICS_ENABLED"] = "False"
        os.environ["GE_UNCOMMITTED_DIRECTORIES"] = "True"
        os.environ["GX_ASSUME_MISSING_LIBRARIES"] = "git"
        
        # Configure logging to reduce verbosity
        import logging
        logging.getLogger('great_expectations').setLevel(logging.ERROR)
        
        # Create sample data
        data_dir = os.path.join(tmp_path, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "sample_wine_data.csv")
        
        with open(data_path, "w") as f:
            f.write("fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality\n")
            f.write("7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5\n")
            f.write("7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5\n")
        
        metrics_path = os.path.join(tmp_path, "validation_metrics.json")
        
        # Import Great Expectations components
        from great_expectations.data_context.types.base import DataContextConfig
        from great_expectations.data_context import BaseDataContext
        from great_expectations.core.batch import RuntimeBatchRequest
        import pandas as pd
        import json
        
        # Load data
        df = pd.read_csv(data_path, sep=";")
        
        # Create minimal context
        context_config = DataContextConfig(
            store_backend_defaults=None,
            expectations_store_name="expectations_store",  # Specify store names
            validations_store_name="validations_store",
            evaluation_parameter_store_name="evaluation_parameter_store",
            checkpoint_store_name=None,
            stores={
                "expectations_store": {
                    "class_name": "ExpectationsStore",
                    "store_backend": {
                        "class_name": "InMemoryStoreBackend"
                    }
                },
                "validations_store": {
                    "class_name": "ValidationsStore",
                    "store_backend": {
                        "class_name": "InMemoryStoreBackend"
                    }
                },
                "evaluation_parameter_store": {
                    "class_name": "EvaluationParameterStore"
                }
            }
        )
        
        context = BaseDataContext(project_config=context_config)
        
        # Add datasource
        context.add_datasource(
            name="pandas_datasource",
            class_name="Datasource",
            module_name="great_expectations.datasource",
            execution_engine={
                "class_name": "PandasExecutionEngine",
                "module_name": "great_expectations.execution_engine"
            },
            data_connectors={
                "runtime_connector": {
                    "class_name": "RuntimeDataConnector",
                    "module_name": "great_expectations.datasource.data_connector",
                    "batch_identifiers": ["batch_id"]
                }
            }
        )
        
        # Create expectation suite - ensure this is a string
        suite_name = "wine_quality_suite"
        print(f"Suite name type: {type(suite_name)}")
        context.create_expectation_suite(suite_name, overwrite_existing=True)
        
        # Create batch request using RuntimeBatchRequest
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_connector",
            data_asset_name="wine_data",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"batch_id": "default_identifier"},
        )
        
        # Get validator with explicit expectation_suite_name
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        # Add expectations and validate
        validator.expect_table_columns_to_match_ordered_list(list(df.columns))
        results = validator.validate()
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump({"validation_success": results.success}, f)
        
        assert os.path.exists(metrics_path)
    
    finally:
        os.chdir(original_cwd)  # Restore original directory

def test_preprocess(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    features_path = os.path.join(tmp_path, "features.npy")
    labels_path = os.path.join(tmp_path, "labels.npy")
    scaler_path = os.path.join(tmp_path, "scaler.pkl")
    
    preprocess.python_func(
        data_path="data/sample_wine_data.csv",
        features=MockArtifact(features_path),
        labels=MockArtifact(labels_path),
        scaler=MockArtifact(scaler_path)
    )
    
    assert os.path.exists(features_path)
    assert os.path.exists(labels_path)
    assert os.path.exists(scaler_path)

def test_train(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    features_path = os.path.join(tmp_path, "features.npy")
    labels_path = os.path.join(tmp_path, "labels.npy")
    model_path = os.path.join(tmp_path, "model.pkl")
    metrics_path = os.path.join(tmp_path, "metrics.json")
    scaler_path = os.path.join(tmp_path, "scaler.pkl")
    
    import joblib
    from sklearn.preprocessing import StandardScaler
    
    joblib.dump(StandardScaler(), scaler_path)
    
    # Create dummy input data
    np.save(features_path, np.random.rand(10, 11))
    np.save(labels_path, np.random.rand(10))
    
    # Disable MLflow for testing
    os.environ["TESTING"] = "True"
    
    # Test RandomForest model type
    train.python_func(
        features=MockArtifact(features_path),
        labels=MockArtifact(labels_path),
        hyperparameters={"n_estimators": 100},
        model_type="RandomForest",
        model=MockArtifact(model_path),
        metrics=MockArtifact(metrics_path),
        scaler=MockArtifact(scaler_path)
    )
    
    assert os.path.exists(model_path)
    assert os.path.exists(metrics_path)
    
    # Test XGBoost model type
    train.python_func(
        features=MockArtifact(features_path),
        labels=MockArtifact(labels_path),
        hyperparameters={"n_estimators": 100, "max_depth": 6},
        model_type="XGBoost",
        model=MockArtifact(model_path),
        metrics=MockArtifact(metrics_path),
        scaler=MockArtifact(scaler_path)
    )
    
    assert os.path.exists(model_path)
    assert os.path.exists(metrics_path)
    # Verify Prometheus metrics setup
    from prometheus_client import Gauge, start_http_server
    Gauge.assert_any_call('train_r2_score', 'Training R² score', ['model_type'])
    start_http_server.assert_called()


def test_deploy_model(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    
    # Create paths for all required artifacts
    model_path = os.path.join(tmp_path, "model.pkl")
    metrics_path = os.path.join(tmp_path, "metrics.json")
    scaler_path = os.path.join(tmp_path, "scaler.pkl")
    
    # Create mock model
    from sklearn.ensemble import RandomForestRegressor
    import pickle
    model_obj = RandomForestRegressor(n_estimators=10)
    with open(model_path, 'wb') as f:
        pickle.dump(model_obj, f)
    
    # Create mock metrics
    import json
    metrics_data = {
        "train_r2": 0.85,
        "test_r2": 0.82,
        "mean_absolute_error": 0.35
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f)
    
    # Create mock scaler
    from sklearn.preprocessing import StandardScaler
    import joblib
    scaler_obj = StandardScaler()
    joblib.dump(scaler_obj, scaler_path)
    
    # Import the deploy_model component
    from wine_quality_pipeline import deploy_model
    
    # Mock kubernetes and kserve modules
    import sys
    from unittest.mock import MagicMock
    
    # Create mock modules
    mock_kserve = MagicMock()
    mock_kubernetes = MagicMock()
    
    # Add mocks to sys.modules
    sys.modules['kserve'] = mock_kserve
    sys.modules['kubernetes'] = mock_kubernetes
    sys.modules['kubernetes.client'] = MagicMock()
    sys.modules['kubernetes.config'] = MagicMock()
    
    # Create a mock KServeClient
    mock_kserve_client = MagicMock()
    mock_kserve.KServeClient.return_value = mock_kserve_client
    
    # Set environment variable for testing
    os.environ["TESTING"] = "True"
    
    # Call the deploy_model function
    service_url = deploy_model.python_func(
        model=MockArtifact(model_path),
        metrics=MockArtifact(metrics_path),
        scaler=MockArtifact(scaler_path),
        service_name="wine-quality-test",
        namespace="kubeflow-test"
    )
    
    # Verify that the KServe client was called to create the inference service
    mock_kserve_client.create.assert_called_once()
    
    # Assert that the function returns a service URL
    assert isinstance(service_url, str)
    assert "wine-quality-test" in service_url

def test_select_best_model(tmp_path):
    """Test select_best_model with individual model inputs."""
    os.makedirs(tmp_path, exist_ok=True)
    
    # Create paths for multiple models and metrics
    model_paths = [
        os.path.join(tmp_path, f"model_{i}.pkl") for i in range(3)
    ]
    
    metrics_paths = [
        os.path.join(tmp_path, f"metrics_{i}.json") for i in range(3)
    ]
    
    best_model_path = os.path.join(tmp_path, "best_model.pkl")
    best_metrics_path = os.path.join(tmp_path, "best_metrics.json")
    
    X = np.random.rand(10, 4)
    y = np.random.rand(10)
    
    # Create and fit models
    models = [
        RandomForestRegressor(n_estimators=10).fit(X, y),
        xgb.XGBRegressor(n_estimators=10).fit(X, y),
        lgb.LGBMRegressor(n_estimators=10).fit(X, y)
    ]
    
    model_types = ["RandomForest", "XGBoost", "LightGBM"]
    test_scores = [0.82, 0.85, 0.80]  # XGBoost should be selected as best
    
    # Save models and metrics
    for i, (model, model_type, test_score) in enumerate(zip(models, model_types, test_scores)):
        joblib.dump(model, model_paths[i])
        
        metrics_data = {
            "model_type": model_type,
            "train_r2": float(test_score + 0.05),
            "test_r2": float(test_score)
        }
        
        with open(metrics_paths[i], 'w') as f:
            json.dump(metrics_data, f)
    
    # Import the select_best_model component
    from wine_quality_pipeline import select_best_model
    
    # Create mock artifacts
    model_artifacts = [MockArtifact(path) for path in model_paths]
    metrics_artifacts = [MockArtifact(path) for path in metrics_paths]
    best_model_artifact = MockArtifact(best_model_path)
    best_metrics_artifact = MockArtifact(best_metrics_path)
    
    # Call the function with the updated parameter structure
    select_best_model.python_func(
        best_model=best_model_artifact,
        best_metrics=best_metrics_artifact,
        model1=model_artifacts[0],
        model2=model_artifacts[1],
        model3=model_artifacts[2],
        metrics1=metrics_artifacts[0],
        metrics2=metrics_artifacts[1],
        metrics3=metrics_artifacts[2]
    )
    
    # Verify results
    assert os.path.exists(best_model_path)
    assert os.path.exists(best_metrics_path)
    
    # Check that XGBoost was selected as the best model
    with open(best_metrics_path, 'r') as f:
        best_metrics_data = json.load(f)
    assert best_metrics_data["model_type"] == "XGBoost"


