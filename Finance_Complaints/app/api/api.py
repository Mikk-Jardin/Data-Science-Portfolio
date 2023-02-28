from fastapi import FastAPI
import tensorflow as tf
from tensorflow import keras
from pydantic import BaseModel
import tensorflow_hub as hub

app = FastAPI(title="Finance Complaint Classifier",
            description="Deep learning model API for classifying the financial product being referred to in a complaint narrative.",
            version="1.0")

model = keras.models.load_model("models/classifier.h5", custom_objects={"KerasLayer": hub.KerasLayer})

class_names = ['Bank account or service', 'Checking or savings account',
        'Consumer Loan', 'Credit card or prepaid card',
        'Credit reporting, credit repair services, or other personal consumer reports',
        'Debt collection',
        'Money transfer, virtual currency, or money service', 'Mortgage',
        'Other financial service',
        'Payday loan, title loan, or personal loan', 'Student loan',
        'Vehicle loan or lease']

class Complaint(BaseModel):
    complaint: str

@app.get("/")
async def home():
    return {"Message": "Welcome to Finance Complaint CLassifier"}

@app.post("/predict")
async def predict(complaint: Complaint):
    to_pred = [complaint.complaint]
    pred_probs = model.predict(to_pred)
    pred = tf.argmax(pred_probs, axis=1)
    return class_names[int(pred)]