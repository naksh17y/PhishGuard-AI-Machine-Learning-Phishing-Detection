from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

# Import our feature extraction logic from our other script!
from feature_engineering import extract_features

# ---------------------------------------------------------
# 1. API Setup & Model Loading
# ---------------------------------------------------------
app = FastAPI(
    title="Phishing Detection API",
    description="An ML-powered API that analyzes URLs for phishing indicators.",
    version="1.0.0"
)

# Allow web pages to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model into memory when the API starts
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'phishing_rf_model.pkl')

if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}. Did you run train_model.py?")

print("Loading ML Model...")
rf_model = joblib.load(model_path)
print("Model loaded successfully!")

# ---------------------------------------------------------
# 2. Data Validation Schema
# ---------------------------------------------------------
# Pydantic ensures the API only accepts valid JSON structures
class URLRequest(BaseModel):
    url: str

# ---------------------------------------------------------
# 3. API Endpoints
# ---------------------------------------------------------
@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "Active", "message": "Phishing Detection API is running."}

@app.post("/predict")
def predict_phishing(request: URLRequest):
    """
    Takes a URL, extracts its features, and runs it through the Random Forest model.
    """
    try:
        # 1. Extract features using the exact same logic we used for training
        # We pass label=0 because we don't know the real label
        raw_features = extract_features(request.url, label=0)
        
        # 2. Remove the 'label' key before feeding it to the model
        del raw_features['label']
        
        # 3. Convert the dictionary into a Pandas DataFrame (which the model expects)
        features_df = pd.DataFrame([raw_features])
        
        # 4. Make the prediction (0 = Benign, 1 = Phishing)
        prediction = rf_model.predict(features_df)[0]
        
        # 5. Get the confidence score (probability)
        probabilities = rf_model.predict_proba(features_df)[0]
        confidence = probabilities[prediction] * 100
        
        # 6. Format the response
        result = "Phishing" if prediction == 1 else "Benign"
        
        return {
            "url": request.url,
            "prediction": result,
            "confidence_score": f"{confidence:.2f}%",
            "extracted_features": raw_features
        }
        
    except Exception as e:
        # Properly catch and display errors
        raise HTTPException(status_code=500, detail=str(e))