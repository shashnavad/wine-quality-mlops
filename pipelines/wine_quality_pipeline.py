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
            
            print("Successfully logged metrics to MLflow")
        except Exception as e:
            print(f"MLflow logging failed (this is expected in local environments): {e}")
            print("Continuing without MLflow logging")
    else:
        print("Testing mode active, skipping MLflow logging")

    print(f"Model saved to {model.path}")
    print(f"Metrics saved to {metrics.path}")
    print(f"Training metrics: {metrics_dict}")
@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def validate_data(
    data_path: str,
    metrics: Output[Metrics]
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
        
        return validation_metrics
        
    except Exception as e:
        print(f"Error in validate_data: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Log failure in metrics
        metrics.log_metric("validation_success", 0.0)
        metrics.log_metric("error", 1.0)
        raise

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

    # Validate data for drift
    validation_task = validate_data(data_path=data_path)
    
    # Preprocess data (only if validation passes)
    preprocess_task = preprocess(data_path=data_path).after(validation_task)
    
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