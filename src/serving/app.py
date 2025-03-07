from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List
import uvicorn

app = FastAPI(title="Wine Quality Prediction API")

# Load the model
model = None
try:
    model = joblib.load("/models/model.joblib")
except:
    # For development, we'll pass if model isn't found
    pass

class WineFeatures(BaseModel):
    features: List[float]

class Prediction(BaseModel):
    quality: float
    confidence: float

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=Prediction)
async def predict(wine: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(wine.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability (confidence)
        # For random forest, we'll use the std of tree predictions
        predictions = np.array([tree.predict(features) for tree in model.estimators_])
        confidence = 1.0 - np.std(predictions) / 10.0  # Normalize to 0-1 scale
        
        return Prediction(quality=float(prediction), confidence=float(confidence))
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 