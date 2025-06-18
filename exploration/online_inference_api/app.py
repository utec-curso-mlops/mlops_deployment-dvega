from fastapi import FastAPI
import numpy as np
import mlflow
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = mlflow.xgboost.load_model("/app/model")

# Get App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Welcome to the ML Model API"}


@app.post("/predict/")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict_proba(features)[:, 1][0]
    return {"prediction": float(prediction)}