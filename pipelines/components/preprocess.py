import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output


@dsl.component(
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
