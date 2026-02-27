from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load the "Brain" files
# VIVA TIP: Ensure these files were exported from the SAME Colab session!
try:
    model = joblib.load("crop_model.joblib")
    le = joblib.load("label_encoder.joblib")
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
    # 1. Strict Feature Alignment
    # This must match X.columns from your Colab exactly.
    feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # 2. Prepare Input
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])[feature_order]
    
    # 3. Model Prediction
    # We use .item() to convert numpy types to standard Python types for JSON
    prediction = model.predict(input_df)
    raw_val = prediction[0]
    
    # 4. Decoding Logic
    try:
        # If it's a number, le.inverse_transform converts it (e.g., 20 -> 'rice')
        crop = le.inverse_transform(prediction)[0]
    except:
        # If the model predicts the string directly
        crop = str(raw_val)
        
    # 5. Return detailed response for debugging
    return {
        "recommended_crop": crop,
        "status": "success",
        "debug_info": {
            "input_received": input_dict,
            "raw_prediction_idx": int(raw_val) if isinstance(raw_val, (int, float, complex)) else str(raw_val)
        },
        "accuracy_used": "99.55%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
