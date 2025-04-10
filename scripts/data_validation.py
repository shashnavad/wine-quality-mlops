import os
import logging
import great_expectations as ge
from great_expectations.data_context import DataContext
from great_expectations.dataset import PandasDataset
import pandas as pd

logger = logging.getLogger(__name__)

def validate_data(input_path):
    """
    Validate data against expectation suite to detect data drift
    
    Args:
        input_path (str): Path to the input data file
    
    Returns:
        dict: Validation results
    """
    try:
        # Initialize Great Expectations context
        context = DataContext(os.path.join(os.getcwd(), "great_expectations"))
        
        # Read the data
        df = pd.read_csv(input_path, sep=';')
        
        # Create a batch of data
        batch = context.get_batch(
            batch_kwargs={"dataset": df, "datasource": "wine_data"},
            expectation_suite_name="wine_quality_suite"
        )
        
        # Run validation
        results = context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[batch],
            run_id="wine_pipeline_run"
        )
        
        # Check if validation passed
        if not results["success"]:
            logger.warning("Data validation failed - possible data drift detected")
        else:
            logger.info("Data validation passed - no drift detected")
            
        return results
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise
        
def alert_data_team(validation_results, alert_method="console"):
    """
    Alert the data team if data drift is detected
    
    Args:
        validation_results (dict): Validation results from Great Expectations
        alert_method (str): Method to use for alerting (console, slack, email)
    """
    if alert_method == "console":
        print("ALERT: Data drift detected. Check validation results.")
    elif alert_method == "slack":
        # Implementation for Slack alerts would go here
        webhook_url = os.getenv("SLACK_WEBHOOK")
        if webhook_url:
            # Use requests to post to Slack
            try:
                import requests
                requests.post(webhook_url, json={"text": "Data drift detected in wine quality pipeline!"})
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {str(e)}")
    # Add more alert methods as needed