from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load the "Brain" files
# Ensure these files are in the same folder as main.py in your GitHub
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
    return {"message": "Crop AI Recommendation API is Live!"}

@app.post("/predict")
def predict(data: SoilData):
    # 1. Force the EXACT order of columns to match the training data in Colab
    # If the order is different, the prediction will be wrong (e.g., showing 'Coffee')
    feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # 2. Convert Pydantic model to dictionary and then to a DataFrame with specific column order
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])[feature_order]
    
    # 3. Make prediction using the Random Forest model
    prediction = model.predict(input_df)
    
    # 4. Handle Label Decoding (converting number back to Crop Name)
    try:
        # If model predicts a number (e.g., 0, 1), we use the label encoder to get 'Rice'
        crop = le.inverse_transform(prediction)[0]
    except Exception:
        # If the model was saved to return strings directly
        crop = str(prediction[0])
        
    # 5. Return the final result to your Flutter App
    return {
        "recommended_crop": crop,
        "status": "success",
        "accuracy_used": "99.55%",
        "order_verified": True
    }

# This allows the API to run if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
