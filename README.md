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
- Write tests for new features

## License

MIT License 
