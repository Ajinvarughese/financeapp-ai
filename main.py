from fastapi import FastAPI
from model import *
from entity import Dataset
from entity import Prompt

app = FastAPI()

@app.post("/ai/risk")
def get_user(dataset: Dataset):
    risk = predict_risk(dataset)
    return risk

@app.post("/ai/save-dataset")
def save_dataset(dataset: Dataset):
    saveDataset(
        dataset.total_assets,
        dataset.total_liabilities,
        dataset.new_liability,
        dataset.income,
        dataset.monthly_emi
    )   

@app.post("/ai/suggestion")
def suggetions(dataset: Dataset):
    pass

@app.post("/ai/chat")
def chat(req: Prompt):
    return {"text": "This is a response from AI"}
