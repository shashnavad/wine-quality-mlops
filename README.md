# Wine Quality MLOps Project

This is an MLOps project that implements an end-to-end machine learning pipeline for wine quality prediction using Kubeflow Pipelines. The project demonstrates MLOps best practices including pipeline orchestration, containerization, and Kubernetes deployment. The pipeline supports multiple model types (RandomForest, XGBoost, LightGBM) with selective training capabilities.

## Project Structure

```
wine-quality-mlops/
├── data/               # Data files and data versioning
├── great_expectations/ 
├── k8s/               # Kubernetes manifests for deployment
├── pipelines/         # Kubeflow/Argo pipeline definitions
│   ├── wine_quality_pipeline.py    # Main ML pipeline
├── scripts/           # Utility scripts
│   ├── run_pipeline.py
│   └── run_server.py
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## Prerequisites

- Kubernetes cluster
- Python 3.8+
- Docker

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Components

1. Data Validation
   - Data quality checks using Great Expectations
   - Schema validation
   - Data drift detection

2. Data Preprocessing
   - Feature engineering
   - Data cleaning
   - Feature scaling

3. Model Training
   - Support for multiple model types:
     - RandomForest
     - XGBoost
     - LightGBM
   - Selective model training via boolean flags
   - Hyperparameter configuration for each model type
   - Automatic model evaluation and comparison

4. Model Evaluation
   - Performance metrics
   - Model validation
   - A/B testing setup

5. Model Serving
   - Model deployment
   - Monitoring setup
   - Inference service

## Usage

1. 1. Start the pipeline with specific model selections:
```bash
python scripts/run_pipeline.py --rf-n-estimators 150 --rf-max-depth 10 --data-path ./data/winequality-red.csv
```
   2. Or run with default settings (all models):

2. Monitor the pipeline in Kubeflow dashboard

3. Access the deployed model:
```bash
curl -X POST http://[MODEL_ENDPOINT]/v1/models/wine-quality:predict -d @sample_input.json
```

## Development

- Use pre-commit hooks for code quality
- Follow the branching strategy


### Project Overview
This is an MLOps project that implements an end-to-end machine learning pipeline for wine quality prediction using Kubeflow Workflows. The project demonstrates MLOps best practices including pipeline orchestration, containerization, and Kubernetes deployment.

### Key Components

1. **ML Pipeline** (`pipelines/wine_quality_pipeline.py`)
   - Data Download & Validation
   - Data Preprocessing with scikit-learn
   - Model Training with multiple algorithms:
     - RandomForest Regressor
     - XGBoost Regressor
     - LightGBM Regressor
   - Automatic Best Model Selection
   - Model Evaluation
   - Model Packaging for Deployment
   - Built using Kubeflow Pipeline SDK v2

2. **Infrastructure**
   - `Dockerfile`: Container definition for the ML components
   - `k8s/`: Kubernetes manifests for deployment

3. **Application Code**
   - `src/`: Core ML functionality
   - `scripts/`:
     - `run_pipeline.py`: Pipeline execution script
     - `run_server.py`: Model serving script

4. **Testing**
```bash
pytest pipelines/test_pipeline.py -v --capture=no --log-cli-level=DEBUG
```
   - `test_model.py`: Model testing functionality

### Pipeline Steps
1. **Data Download**
   - Downloads wine quality dataset from UCI repository
   - Saves to temporary storage

2. **Data Validation**
   - Checks for missing values
   - Validates quality range
   - Generates validation report

3. **Preprocessing**
   - Feature scaling using StandardScaler
   - Data splitting for training/testing
   - Saves preprocessed features and labels

4. **Model Training**
   - Selectively trains models based on boolean flags:
     - RandomForest
     - XGBoost
     - LightGBM
   - Configurable hyperparameters for each model type
   - Models are trained independently based on user selection

5. **Model Evaluation**
   - Calculates metrics:
     - MSE
     - RMSE
     - MAE
     - R² score

6. **Best Model Selection**
   - Automatically compares all trained models
   - Selects the model with the highest test R² score
   - Generates comparison metrics between models

7. **Model Deployment Packaging**
   - Bundles model artifacts
   - Includes:
     - Trained model with the best metrics
     - Scaler
     - Metrics
     - API specifications

## License

MIT License 

## Running the Pipeline in Kubernetes

Since setting up Kubeflow Pipelines UI can be challenging in some environments, I've provided a simpler approach to run the pipeline directly in Kubernetes using Jobs.

### Using the Simple Job

To run the pipeline with default hyperparameters:

```bash
kubectl apply -f k8s/pipeline-job.yaml
```

### Using the Parameterized Job

To run the pipeline with custom model selection and hyperparameters:

1. Edit the environment variables in `k8s/parameterized-job.yaml` to include:
   - USE_RANDOM_FOREST: "true" or "false"
   - USE_XGBOOST: "true" or "false"
   - USE_LIGHTGBM: "true" or "false"
   - RF_N_ESTIMATORS: "150"
   - RF_MAX_DEPTH: "10"
   - And other model-specific parameters

2. Apply the job:

```bash
kubectl apply -f k8s/parameterized-job.yaml
```

### Viewing Results in MLflow

The pipeline logs metrics and models to MLflow. To access the MLflow UI:

```bash
kubectl port-forward -n mlops svc/mlflow-service 5000:5000
```

Then open http://localhost:5000 in your browser. 

## MLflow and MinIO Integration

To ensure MLflow can store artifacts properly, we need to set up a MinIO bucket:

1. Create a Kubernetes secret for MinIO credentials:

```bash
kubectl create secret generic mlflow-minio-credentials -n mlops \
  --from-literal=AWS_ACCESS_KEY_ID=minioadmin \
  --from-literal=AWS_SECRET_ACCESS_KEY=minioadmin
```

2. Create the MinIO bucket for MLflow:

```bash
kubectl apply -f k8s/create-minio-bucket.yaml
```

This will create a bucket named "mlflow" in MinIO that MLflow will use to store model artifacts. 


## Why the Kubeflow/MLFlow Setup Works

The pipeline can run on MLflow even with Kubeflow-specific files because:

1. **Independent Execution**: The Python code in the components can run independently of Kubeflow
2. **MLflow Integration**: The components include MLflow logging code
3. **Kubernetes Deployment**: I've deployed MLflow on Kubernetes, which my pipeline connects to

The `wine_quality_pipeline.yaml` file is only used when you want to run the pipeline on Kubeflow. When running with MLflow, I'm executing a Python script directly that:

1. Imports the component functions
2. Calls them in sequence
3. Passes data between them
4. Logs results to MLflow

