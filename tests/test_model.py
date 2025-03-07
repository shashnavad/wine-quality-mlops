#!/usr/bin/env python3
"""
Script to test the wine quality model locally.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from src.data.data_loader import WineDataLoader
from src.models.model_trainer import ModelTrainer
from src.serving.model_server import ModelServer, WineFeatures


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Wine Quality model locally")
    
    parser.add_argument("--train", action="store_true",
                        help="Train a new model before testing")
    parser.add_argument("--model-path", type=str, default="models/random_forest_model.joblib",
                        help="Path to the trained model")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of test samples to use")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def train_model():
    """Train a new model."""
    print("Training a new model...")
    trainer = ModelTrainer()
    model, metrics = trainer.train_model(log_to_mlflow=False)
    
    print("\nModel training complete.")
    print("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return model


def get_test_samples(n_samples, random_seed):
    """Get random test samples from the dataset."""
    print(f"\nGetting {n_samples} test samples...")
    
    # Load the data
    loader = WineDataLoader()
    df = loader.load_data()
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Select random samples
    sample_indices = np.random.choice(len(df), size=n_samples, replace=False)
    samples = df.iloc[sample_indices]
    
    return samples


def test_model(model_server, samples):
    """Test the model on the given samples."""
    print("\nTesting model on samples:")
    
    for i, (_, sample) in enumerate(samples.iterrows(), 1):
        # Create input features
        wine_features = WineFeatures(
            fixed_acidity=float(sample['fixed acidity']),
            volatile_acidity=float(sample['volatile acidity']),
            citric_acid=float(sample['citric acid']),
            residual_sugar=float(sample['residual sugar']),
            chlorides=float(sample['chlorides']),
            free_sulfur_dioxide=float(sample['free sulfur dioxide']),
            total_sulfur_dioxide=float(sample['total sulfur dioxide']),
            density=float(sample['density']),
            pH=float(sample['pH']),
            sulphates=float(sample['sulphates']),
            alcohol=float(sample['alcohol'])
        )
        
        # Convert to numpy array
        features = np.array([
            wine_features.fixed_acidity,
            wine_features.volatile_acidity,
            wine_features.citric_acid,
            wine_features.residual_sugar,
            wine_features.chlorides,
            wine_features.free_sulfur_dioxide,
            wine_features.total_sulfur_dioxide,
            wine_features.density,
            wine_features.pH,
            wine_features.sulphates,
            wine_features.alcohol
        ])
        
        # Make prediction
        result = model_server.predict(features)
        
        # Print results
        print(f"\nSample {i}:")
        print(f"  Actual quality: {sample['quality']}")
        print(f"  Predicted quality: {result['quality']:.2f}")
        if result['confidence'] is not None:
            print(f"  Confidence: {result['confidence']:.2f}")
        
        # Print feature values
        print("  Features:")
        for field_name, field_value in wine_features.dict().items():
            print(f"    {field_name}: {field_value}")


def main():
    """Main function to test the model."""
    args = parse_arguments()
    
    # Train a new model if requested
    if args.train:
        train_model()
    
    # Initialize model server
    model_server = ModelServer(model_path=args.model_path)
    
    # Get test samples
    samples = get_test_samples(args.n_samples, args.random_seed)
    
    # Test the model
    test_model(model_server, samples)


if __name__ == "__main__":
    main() 