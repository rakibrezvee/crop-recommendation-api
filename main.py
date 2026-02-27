from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# Load model directly
MODEL_PATH = "crop_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Model load failed: {e}")

class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def home():
    return {
        "status": "online",
        "model_ready": model_loaded,
        "files": os.listdir('.')
    }

@app.post("/predict")
def predict(data: SoilData):
    if model is None:
        return {"error": "Model not loaded", "status": 500}
    
    try:
        # 1. Align features
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([data.dict()])[feature_order]
        
        # 2. Predict index
        prediction = model.predict(input_df)
        raw_idx = int(prediction[0])
        
        # 3. Hardcoded Map (Corrected alphabetical order)
        crops = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 
            'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 
            'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 
            'muskmelon', 'orange', 'papaya', 'pigeonpeas', 
            'pomegranate', 'rice', 'watermelon'
        ]
        
        crop_name = crops[raw_idx] if raw_idx < len(crops) else "Unknown"
        
        return {
            "recommended_crop": crop_name,
            "raw_index": raw_idx,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": 500}
