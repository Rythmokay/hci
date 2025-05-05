from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pickle
import os
import numpy as np
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Import the training module
from app.model.train import train_model

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction")

# Configure templates and static files
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Check if model exists, if not train it
model_path = Path("app/model/model.pkl")
if not model_path.exists():
    # Make sure the directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = train_model()
else:
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

class HouseData(BaseModel):
    sqft: float
    bedrooms: int
    bathrooms: int
    age: int

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    sqft: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    age: int = Form(...)
):
    # Create features array for prediction
    features = np.array([[sqft, bedrooms, bathrooms, age]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Format the prediction as currency
    formatted_prediction = f"${prediction:,.2f}"
    
    return {"prediction": formatted_prediction}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)