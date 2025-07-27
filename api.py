from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

app = FastAPI()

xgbModel = joblib.load('xgb_model_new.pkl')
scaler = joblib.load('scaler_new.pkl')


class HouseInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float 
    households: float 
    median_income: float
    ocean_proximity: int 
    #population_per_household: float
    #rooms_per_household: float
    #bedrooms_per_room: float = Field(..., alias="bedrooms_per_room")
   
    
    

    #ocean_proximity_1H_OCEAN: bool = Field(..., alias="ocean_proximity_<1H OCEAN")
    #ocean_proximity_INLAND: bool
    #ocean_proximity_ISLAND: bool
    #ocean_proximity_NEAR_BAY: bool = Field(..., alias="ocean_proximity_NEAR BAY")
    #ocean_proximity_NEAR_OCEAN: bool = Field(..., alias="ocean_proximity_NEAR OCEAN")


@app.post("/predict")
async def predictValue(input: HouseInput):
    input_dict = input.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    pred_log = xgbModel.predict(input_scaled)
    prediction = np.expm1(pred_log)
    return {"model": "xgboost", "predicted_value": float(prediction)}
