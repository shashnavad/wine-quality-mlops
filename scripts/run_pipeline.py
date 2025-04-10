#!/usr/bin/env python3
"""
Script to run the Wine Quality MLOps pipeline.
"""

import os
import argparse
import kfp
from pipelines.wine_quality_pipeline import wine_quality_pipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Wine Quality MLOps pipeline")
    
    # Pipeline parameters
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees in the random forest")
    parser.add_argument("--max-depth", type=str, default="None",
                        help="Maximum depth of the trees. Use 'None' for unlimited depth")
    parser.add_argument("--min-samples-split", type=int, default=2,
                        help="Minimum number of samples required to split an internal node")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                        help="Minimum number of samples required to be at a leaf node")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random state for reproducibility")
    
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
    
    # Run the pipeline
    run_name = args.run_name or f"wine-quality-{args.n_estimators}-trees"
    print(f"Running pipeline in experiment '{args.experiment_name}' with run name '{run_name}'...")
    
    # Set up the pipeline parameters
    params = {
        "data_path": args.data_path,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state
    }
    
    # Run the pipeline
    run = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=run_name,
        pipeline_package_path=pipeline_file,
        params=params
    )
    
    print(f"Pipeline run started with ID: {run.id}")
    print(f"View the run at: {args.host}/#/runs/details/{run.id}")


if __name__ == "__main__":
    main() 