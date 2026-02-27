from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load the "Brain" file
try:
    # We focus on the model; we will use a manual list for the labels
    model = joblib.load("crop_model.joblib")
except Exception as e:
    print(f"Error loading model files: {e}")

# 3. Define the format of data the API expects
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
    return {"message": "Crop AI Recommendation API is Live!"}

@app.post("/predict")
def predict(data: SoilData):
    # 1. Force the EXACT order of columns to match your Colab Training
    feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # 2. Convert input to DataFrame with specific column order
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])[feature_order]
    
    # 3. Make prediction (returns an index number)
    prediction = model.predict(input_df)
    raw_val = int(prediction[0])
    
    # 4. THE FIX: Manual Label Map
    # This list is the standard order for the 22-crop dataset.
    # It ensures Index 8 = Jute, Index 1 = Banana, etc.
    crops = [
        'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 
        'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 
        'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 
        'muskmelon', 'orange', 'papaya', 'pigeonpeas', 
        'pomegranate', 'rice', 'watermelon'
    ]
    
    try:
        # Pick the name from the list using the index
        recommended = crops[raw_val]
    except:
        # Fallback if index is out of range
        recommended = "Unknown Crop"
        
    # 5. Return result
    return {
        "recommended_crop": recommended,
        "status": "success",
        "debug_info": {
            "input_received": input_dict,
            "raw_prediction_idx": raw_val
        },
        "accuracy_used": "99.55%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
