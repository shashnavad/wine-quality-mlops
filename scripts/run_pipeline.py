#!/usr/bin/env python3

"""
Script to run the Wine Quality MLOps pipeline.
"""

import os
import sys
import argparse
import kfp

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipelines.wine_quality_pipeline import wine_quality_pipeline

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Wine Quality MLOps pipeline")
    
    # Model selection parameters
    parser.add_argument("--use-random-forest", action="store_true", default=True,
                    help="Train RandomForest model")
    parser.add_argument("--use-xgboost", action="store_true", default=True,
                        help="Train LightGBM model")
    parser.add_argument("--use-lightgbm", action="store_true", default=True,
                        help="Train LightGBM model")
    
    # RandomForest parameters
    parser.add_argument("--rf-n-estimators", type=int, default=100,
                        help="Number of trees in the random forest")
    parser.add_argument("--rf-max-depth", type=str, default="None",
                        help="Maximum depth of the RF trees. Use 'None' for unlimited depth")
    parser.add_argument("--rf-min-samples-split", type=int, default=2,
                        help="Minimum number of samples required to split an internal node in RF")
    
    # XGBoost parameters
    parser.add_argument("--xgb-n-estimators", type=int, default=100,
                        help="Number of trees in XGBoost")
    parser.add_argument("--xgb-max-depth", type=int, default=6,
                        help="Maximum depth of the XGBoost trees")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1,
                        help="Learning rate for XGBoost")
    
    # LightGBM parameters
    parser.add_argument("--lgbm-n-estimators", type=int, default=100,
                        help="Number of trees in LightGBM")
    parser.add_argument("--lgbm-max-depth", type=int, default=-1,
                        help="Maximum depth of the LightGBM trees. -1 means no limit")
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.1,
                        help="Learning rate for LightGBM")
    
    # Kubeflow options
    parser.add_argument("--host", type=str, default="http://localhost:8080",
                        help="Kubeflow Pipelines API host")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile the pipeline, don't run it")
    parser.add_argument("--output-file", type=str, default="wine_quality_pipeline.yaml",
                        help="Output file for the compiled pipeline")
    parser.add_argument("--experiment-name", type=str, default="wine-quality",
                        help="Experiment name in Kubeflow")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name in Kubeflow (default: auto-generated)")
    parser.add_argument("--data-path", type=str,
                        default="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                        help="Path to the wine quality data file")
    parser.add_argument("--service-name", type=str, default="wine-quality-predictor",
                        help="Name of the deployed service")
                        
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()

    # Compile the pipeline
    pipeline_file = args.output_file
    print(f"Compiling pipeline to {pipeline_file}...")
    kfp.compiler.Compiler().compile(wine_quality_pipeline, pipeline_file)
    print(f"Pipeline compiled successfully to {pipeline_file}")

    if args.compile_only:
        return

    # Create the Kubeflow Pipelines client
    client = kfp.Client(host=args.host)

    # Create or get the experiment
    experiment = client.create_experiment(args.experiment_name)

    # Convert string "None" to Python None for max_depth
    rf_max_depth = None if args.rf_max_depth.lower() == "none" else int(args.rf_max_depth)

    # Set up the pipeline parameters
    params = {
        "data_path": args.data_path,
        "use_random_forest": args.use_random_forest,
        "use_xgboost": args.use_xgboost,
        "use_lightgbm": args.use_lightgbm,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": rf_max_depth,
        "rf_min_samples_split": args.rf_min_samples_split,
        "xgb_n_estimators": args.xgb_n_estimators,
        "xgb_max_depth": args.xgb_max_depth,
        "xgb_learning_rate": args.xgb_learning_rate,
        "lgbm_n_estimators": args.lgbm_n_estimators,
        "lgbm_max_depth": args.lgbm_max_depth,
        "lgbm_learning_rate": args.lgbm_learning_rate,
        "service_name": args.service_name
    }

    # Run the pipeline
    run = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=args.run_name,
        pipeline_package_path=pipeline_file,
        params=params
    )

    print(f"Pipeline run started with ID: {run.id}")
    print(f"View the run at: {args.host}/#/runs/details/{run.id}")

if __name__ == "__main__":
    main()