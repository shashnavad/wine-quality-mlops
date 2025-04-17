import os
import sys
import json
import pandas as pd
import logging
from kfp import dsl
from kfp.dsl import component, Output, Metrics

# Great Expectations imports
from great_expectations.data_context.types.base import DataContextConfig
from great_expectations.data_context import BaseDataContext
from great_expectations.core.batch import RuntimeBatchRequest

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

@dsl.component(
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