from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


#Load the saved papeline once when the server starts
model = joblib.load("churn_pipeline.pkl")
#Using tuned threshold
THRESHOLD = 0.4
app = FastAPI(title="Customer Churn Prediction API")

#Define the input schema
class CustomerInput(BaseModel):
    gender:str
    SeniorCitizen:int
    Partner:str
    Dependents:str
    tenure:int
    PhoneService:str
    MultipleLines:str
    InternetService :str
    OnlineSecurity :str
    OnlineBackup : str
    DeviceProtection : str
    TechSupport :str
    StreamingTV :str
    StreamingMovies :  str
    Contract :  str
    PaperlessBilling : str
    PaymentMethod : str
    MonthlyCharges : float
    TotalCharges :float

#Health Check route (To confirm server is running)
@app.get("/")
def home():
    return{"status": "OK", "message": "Churn model API is running"}

#prediction route
@app.post("/predict")
def predict_churn(data: CustomerInput):
    #Convert incoming JSON into a 1-row pandas Dataframe
    df = pd.DataFrame([data.model_dump()]) #pydantic v2 method
    #predict Churn probability( Class = 1)
    proba = float(model.predict_proba(df)[:,1][0])

    #Applying tuned threshold(0.4)
    pred = int(proba >= THRESHOLD)
    return{
        "churn_probability": proba,
        "predicted_churn": pred,
        "threshold_used": THRESHOLD
    }