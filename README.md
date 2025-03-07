# Wine Quality MLOps Project

This project demonstrates MLOps best practices using Kubeflow pipelines for wine quality prediction. It implements a full ML pipeline including data validation, preprocessing, training, evaluation, and model serving.

## Project Structure

```
wine-quality-mlops/
├── data/               # Data files and data versioning
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── k8s/               # Kubernetes manifests for deployment
├── models/            # Trained models and artifacts
│   ├── trained/       # Saved model files
│   └── configs/       # Model configurations
├── pipelines/         # Kubeflow/Argo pipeline definitions
│   ├── wine_quality_pipeline.py    # Main ML pipeline
├── scripts/           # Utility scripts
│   ├── run_pipeline.py
│   └── run_server.py
├── src/              # Source code for ML components
├── tests/            # Test files
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## Prerequisites

- Kubernetes cluster
- Kubeflow installed
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

3. Configure Kubeflow:
- Ensure Kubeflow is installed in your cluster
- Configure access to your Kubeflow dashboard

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
   - Hyperparameter tuning
   - Cross-validation
   - Model selection

4. Model Evaluation
   - Performance metrics
   - Model validation
   - A/B testing setup

5. Model Serving
   - Model deployment
   - Monitoring setup
   - Inference service

## Usage

1. Start the pipeline:
```bash
python src/pipelines/wine_quality_pipeline.py
```

2. Monitor the pipeline in Kubeflow dashboard

3. Access the deployed model:
```bash
curl -X POST http://[MODEL_ENDPOINT]/v1/models/wine-quality:predict -d @sample_input.json
```

## Development

- Use pre-commit hooks for code quality
- Follow the branching strategy


### Project Overview
This is an MLOps project that implements an end-to-end machine learning pipeline for wine quality prediction using Kubeflow/Argo Workflows. The project demonstrates MLOps best practices including pipeline orchestration, containerization, and Kubernetes deployment.

### Key Components

1. **ML Pipeline** (`pipelines/wine_quality_pipeline.py`)
   - Data Download & Validation
   - Data Preprocessing with scikit-learn
   - Model Training (Random Forest Regressor)
   - Model Evaluation
   - Model Packaging for Deployment
   - Built using Kubeflow Pipeline SDK

2. **Infrastructure**
   - `Dockerfile`: Container definition for the ML components
   - `k8s/`: Kubernetes manifests for deployment
   - Currently running on Minikube with Argo Workflows

3. **Data Management** (`data/`)
   - Structured with `raw/` and `processed/` directories
   - Uses the UCI Wine Quality dataset
   - Includes data validation checks

4. **Model Management** (`models/`)
   - `trained/`: Directory for saved models
   - `configs/`: Model configuration files
   - Uses scikit-learn's Random Forest implementation

5. **Application Code**
   - `src/`: Core ML functionality
   - `scripts/`:
     - `run_pipeline.py`: Pipeline execution script
     - `run_server.py`: Model serving script

6. **Testing** (`tests/`)
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
   - Trains Random Forest model
   - Configurable hyperparameters:
     - n_estimators
     - max_depth
     - min_samples_split
     - min_samples_leaf

5. **Model Evaluation**
   - Calculates metrics:
     - MSE
     - RMSE
     - MAE
     - R² score

6. **Model Deployment Packaging**
   - Bundles model artifacts
   - Includes:
     - Trained model
     - Scaler
     - Metrics
     - API specifications

### Current Status
- Basic project structure is set up
- Pipeline code is implemented and version controlled
- Argo Workflows is installed on Minikube
- Working on deploying the pipeline in the lightweight Argo environment

### Next Steps (Potential)
1. Complete Argo Workflows integration
2. Add monitoring and logging
3. Implement CI/CD pipeline
4. Add model versioning
5. Enhance testing coverage

Would you like me to elaborate on any specific component or provide more details about a particular aspect of the project?

- Write tests for new features

## License

MIT License 
