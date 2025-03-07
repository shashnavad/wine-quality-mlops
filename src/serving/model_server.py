import os
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import sys

# Add the parent directory to the path to import from src.models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.model_trainer import ModelTrainer


class WineFeatures(BaseModel):
    """Input features for wine quality prediction."""
    
    fixed_acidity: float = Field(..., description="Fixed acidity")
    volatile_acidity: float = Field(..., description="Volatile acidity")
    citric_acid: float = Field(..., description="Citric acid")
    residual_sugar: float = Field(..., description="Residual sugar")
    chlorides: float = Field(..., description="Chlorides")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide")
    density: float = Field(..., description="Density")
    pH: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates")
    alcohol: float = Field(..., description="Alcohol")
    
    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }


class WineFeaturesArray(BaseModel):
    """Input features as array for wine quality prediction."""
    
    features: List[float] = Field(..., description="Array of wine features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
            }
        }


class Prediction(BaseModel):
    """Prediction output."""
    
    quality: float = Field(..., description="Predicted wine quality")
    confidence: Optional[float] = Field(None, description="Prediction confidence")


class ModelServer:
    """Model server for wine quality prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the model server.
        
        Args:
            model_path: Path to the trained model. If None, use the default model.
        """
        if model_path is None:
            model_path = os.path.join("models", "random_forest_model.joblib")
        
        self.model = None
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make a prediction.
        
        Args:
            features: Input features.
        
        Returns:
            Dictionary with prediction and confidence.
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Reshape features if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get prediction confidence (for random forest, use std of tree predictions)
        confidence = None
        if hasattr(self.model, "estimators_"):
            predictions = np.array([tree.predict(features) for tree in self.model.estimators_])
            confidence = 1.0 - np.std(predictions) / 10.0  # Normalize to 0-1 scale
        
        return {
            "quality": float(prediction),
            "confidence": float(confidence) if confidence is not None else None
        }


# Create FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality",
    version="1.0.0",
)

# Initialize model server
model_server = ModelServer()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Wine Quality Prediction API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=Prediction)
async def predict(wine: WineFeatures):
    """Predict wine quality from features."""
    try:
        features = np.array([
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.citric_acid,
            wine.residual_sugar,
            wine.chlorides,
            wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.pH,
            wine.sulphates,
            wine.alcohol
        ])
        
        result = model_server.predict(features)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_array", response_model=Prediction)
async def predict_array(wine: WineFeaturesArray):
    """Predict wine quality from feature array."""
    try:
        features = np.array(wine.features)
        
        if len(features) != 11:
            raise ValueError("Expected 11 features")
        
        result = model_server.predict(features)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("model_server:app", host="0.0.0.0", port=8080, reload=True) 