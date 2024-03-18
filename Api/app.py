from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.responses import PlainTextResponse

# Load models from .pkl files
with open('linear_regression_model.pkl', 'rb') as file:
    lin_reg = pickle.load(file)

with open('gradient_boosting_model.pkl', 'rb') as file:
    grb_reg = pickle.load(file)

# Define input data model
class InputData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# Initialize FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    message ="Welcome to the Insurance Charges Prediction API!\n" \
              "Make a POST request to /predict with the following data in the request body: 'age', 'sex', 'bmi', 'children', 'smoker', 'region'.\n" \
              "For API documentation, visit: http://localhost:8000/docs"

    return PlainTextResponse(content=message, status_code=200)


# Define endpoint for making predictions
@app.post("/predict/")
async def predict_charges(input_data: InputData):
    # Map categorical variables to numerical values
    input_data.sex = 1 if input_data.sex.lower() == "male" else 0
    input_data.smoker = 1 if input_data.smoker.lower() == "yes" else 0
    region_mapping = {"southeast": 0, "southwest": 1, "northwest": 2, "northeast": 3}
    input_data.region = region_mapping[input_data.region.lower()]

    # Convert input data to numpy array
    features = np.array([[input_data.age, input_data.sex, input_data.bmi, input_data.children, input_data.smoker, input_data.region]])

    # Make predictions using both models
    lin_reg_prediction = lin_reg.predict(features)
    grb_reg_prediction = grb_reg.predict(features)

    # Return predictions
    return {"Linear Regression Prediction": float(lin_reg_prediction[0]), "Gradient Boosting Prediction": float(grb_reg_prediction[0])}
