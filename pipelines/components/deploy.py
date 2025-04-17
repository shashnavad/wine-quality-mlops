import os
import json
import pickle
import tempfile
from kfp import dsl
from kfp.dsl import component, Input, Output, Model, Metrics

# KServe and Kubernetes imports
from kubernetes import client
from kubernetes import config
from kserve import KServeClient
from kserve import constants
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1SKLearnSpec

@dsl.component(
    base_image='pes1ug19cs601/wine-quality-mlops:latest',
    packages_to_install=['kserve==0.10.1']
)
def deploy_model(
    model: Input[Model],
    metrics: Input[Metrics],
    scaler: Input[Model],
    service_name: str,
    namespace: str = "kubeflow"
) -> str:
    import os
    import json
    import pickle
    import tempfile
    from kubernetes import client
    from kubernetes import config
    from kserve import KServeClient
    from kserve import constants
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1SKLearnSpec
    
    # Create a temporary directory to prepare the model
    model_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(model_dir, "model"), exist_ok=True)
    
    # Load and save the model and scaler
    with open(model.path, 'rb') as f:
        model_obj = pickle.load(f)
        
    with open(scaler.path, 'rb') as f:
        scaler_obj = pickle.load(f)
    
    # Save model and scaler to the temporary directory
    with open(os.path.join(model_dir, "model", "model.pkl"), 'wb') as f:
        pickle.dump(model_obj, f)
        
    with open(os.path.join(model_dir, "model", "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler_obj, f)
    
    # Load metrics to include in model metadata
    with open(metrics.path, 'r') as f:
        metrics_data = json.load(f)
    
    # Create a metadata file
    metadata = {
        "name": service_name,
        "version": "v1",
        "metrics": metrics_data
    }
    
    with open(os.path.join(model_dir, "model", "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    # Create a simple inference script
    inference_script = """
import os
import pickle
import json
import numpy as np

class WineQualityModel(object):
    def __init__(self):
        self.model = None
        self.scaler = None
        self.ready = False
        
    def load(self):
        model_dir = os.path.join(os.getcwd(), "model")
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        self.ready = True
        
    def predict(self, X, feature_names=None):
        if not self.ready:
            self.load()
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions.tolist()
"""
    
    with open(os.path.join(model_dir, "model", "WineQualityModel.py"), 'w') as f:
        f.write(inference_script)
    
    # Create a simple requirements.txt
    with open(os.path.join(model_dir, "model", "requirements.txt"), 'w') as f:
        f.write("scikit-learn==1.0.2\nnumpy==1.22.3\n")
    
    # Upload the model to a storage location (MinIO, S3, etc.)
    # For this example, we'll assume you have a PVC for model storage
    model_uri = f"pvc://{service_name}-models"
    
    # In a real implementation, you would upload the model to your storage
    # For now, we'll just print the path and assume it's accessible
    print(f"Model prepared at: {model_dir}")
    print(f"Model would be deployed from: {model_uri}")
    
    try:
        # Initialize KServe client
        config.load_incluster_config()
        kserve_client = KServeClient()
        
        # Define the inference service
        isvc = V1beta1InferenceService(
            api_version=constants.KSERVE_V1BETA1,
            kind=constants.KSERVE_KIND,
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=namespace,
                annotations={"sidecar.istio.io/inject": "false"}
            ),
            spec=V1beta1InferenceServiceSpec(
                predictor=V1beta1PredictorSpec(
                    sklearn=V1beta1SKLearnSpec(
                        storage_uri=model_uri,
                        resources=client.V1ResourceRequirements(
                            requests={"cpu": "100m", "memory": "1Gi"},
                            limits={"cpu": "1", "memory": "2Gi"}
                        )
                    )
                )
            )
        )
        
        # Create the inference service
        kserve_client.create(isvc)
        print(f"Inference service '{service_name}' created in namespace '{namespace}'")
        
        # Get the URL of the deployed service
        service_url = f"http://{service_name}.{namespace}.svc.cluster.local/v1/models/{service_name}:predict"
        return service_url
        
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        return f"Deployment failed: {str(e)}"