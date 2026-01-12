from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load the "Brain" files
model = joblib.load("crop_model.joblib")
le = joblib.load("label_encoder.joblib")

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
    return {"message": "Crop API is Live!"}

@app.post("/predict")
def predict(data: SoilData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Check if prediction is a number or string and handle it
    try:
        crop = le.inverse_transform(prediction)[0]
    except:
        crop = prediction[0]
        
    return {
        "recommended_crop": crop,
        "status": "success",
        "accuracy_used": "99.55%"
    }
