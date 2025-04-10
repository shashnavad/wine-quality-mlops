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

@component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest'
)
def validate_data(
    data_path: str,
    metrics: Output[Metrics]
):
    """Validate wine data for drift using Great Expectations."""
    import os
    import sys
    import json
    import pandas as pd
    from great_expectations.data_context import DataContext
    # Disable Git usage in Great Expectations
    os.environ["GE_USAGE_STATISTICS_ENABLED"] = "False"

    
    # Try to initialize Great Expectations context
    try:
        # Check if great_expectations directory exists in the current directory
        if not os.path.exists('great_expectations'):
            from great_expectations.data_context.types.base import DataContextConfig
            from great_expectations.data_context import BaseDataContext
            
            # Create minimal config for in-memory context
            context_config = DataContextConfig(
                store_backend_defaults=None,
                checkpoint_store_name=None,
            )
            context = BaseDataContext(project_config=context_config)
            
            # Create simple expectation suite programmatically
            suite = context.create_expectation_suite("wine_quality_suite")
            
            # Load the data to create expectations
            df = pd.read_csv(data_path, sep=";")
            batch = context.get_batch({}, batch_data=df)
            
            # Add basic expectations for wine data
            batch.expect_table_columns_to_match_ordered_list(
                list(df.columns),
                result_format="COMPLETE"
            )
            batch.expect_column_values_to_be_between("fixed acidity", 4.0, 16.0)
            batch.expect_column_values_to_be_between("volatile acidity", 0.1, 1.2)
            batch.expect_column_values_to_be_between("pH", 2.7, 4.0)
            batch.expect_column_values_to_be_between("alcohol", 8.0, 15.0)
            batch.expect_column_values_to_be_between("quality", 3, 9)
            batch.expect_column_mean_to_be_between("alcohol", 10.0, 11.5)
            
            # Save expectation suite
            context.save_expectation_suite(suite)
        else:
            # Use existing context
            context = DataContext()
        
        # Load new data for validation
        df = pd.read_csv(data_path, sep=";")
        
       # Create batch and validate using V3 API
        data_source = context.sources.add_pandas_source("wine_source")
        data_asset = data_source.add_dataframe_asset("wine_data")
        batch_request = data_asset.build_batch_request(dataframe=df)
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name="wine_quality_suite"
        )
        results = validator.validate()

        
        # Process validation results
        validation_success = results["success"]
        expectations_results = results["results"][0]["result"]
        
        # Count successful and failed expectations
        expectations_total = len(expectations_results["results"])
        expectations_success = sum(1 for r in expectations_results["results"] if r["success"])
        
        # Prepare metrics
        metrics_dict = {
            "validation_success": validation_success,
            "expectations_total": expectations_total,
            "expectations_passed": expectations_success,
            "success_rate": expectations_success / expectations_total if expectations_total > 0 else 0
        }
        
        # Save metrics
        with open(metrics.path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        print(f"Data validation completed with success: {validation_success}")
        print(f"Passed {expectations_success} out of {expectations_total} expectations")
        
        # Optional: fail pipeline if validation fails
        if not validation_success:
            print("WARNING: Data drift detected! Check validation results.")
            # Uncomment to make pipeline fail on drift detection
            # raise ValueError("Data validation failed - possible data drift detected")
        
    except Exception as e:
        print(f"Error during data validation: {str(e)}")
        # Save error in metrics
        with open(metrics.path, 'w') as f:
            json.dump({"error": str(e)}, f)
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