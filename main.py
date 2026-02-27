from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Global variable for the model
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # DOUBLE CHECK THIS FILENAME IN GITHUB!
        model = joblib.load("crop_model.joblib")
    except Exception as e:
        print(f"LOAD ERROR: {e}")

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
    return {"status": "online"}

@app.post("/predict")
def predict(data: SoilData):
    if model is None:
        return {"error": "Model not loaded on server", "status": 500}
    
    try:
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([data.dict()])[feature_order]
        
        prediction = model.predict(input_df)
        raw_val = int(prediction[0])
        
        crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
        
        return {
            "recommended_crop": crops[raw_val] if raw_val < len(crops) else "Unknown",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": 500}
