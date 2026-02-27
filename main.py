from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# 1. Load the model directly
# If this file name is different in your GitHub, CHANGE IT HERE
MODEL_PATH = "crop_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Model load failed: {e}")

# 2. Define the input structure
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
        "model_ready": model is not None,
        "files_in_folder": os.listdir('.')
    }

@app.post("/predict")
def predict(data: SoilData):
    if model is None:
        return {"error": "Model file not found on server", "status": 500}
    
    try:
        # 3. Force the exact column order used in training
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([data.dict()])[feature_order]
        
        # 4. Predict
        prediction = model.predict(input_df)
        raw_idx = int(prediction[0])
        
        # 5. The Hardcoded Crop List (Ensures Jute/Banana match Colab)
        crops = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 
            'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 
            'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 
            'muskmelon', 'orange', 'papaya', 'pigeonpeas', 
            'pomegranate', 'rice', 'watermelon'
        ]
        
        result = crops[raw_idx] if raw_idx < len(crops) else "Unknown"
        
        return {
            "recommended_crop": result,
            "status": "success",
            "debug": {"index": raw_idx}
        }
    except Exception as e:
        return {"error": str(e), "status": 500}
