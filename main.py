from fastapi import FastAPI, UploadFile, File
from model import *
from entity import Dataset
from entity import ChatRequest
from chat_bot import askAI
from fastapi.responses import PlainTextResponse

app = FastAPI()

@app.post("/ai/chat", response_class=PlainTextResponse)
def chat(req: ChatRequest):
    return askAI(
        req.prompt,
        req.chatLog,
        asset=req.assets,
        liability=req.liability,
        user=req.user
    )


app = FastAPI()

# @app.post("/ai/risk")
# def predict(asset: Asset, liabilities: Liability, new_liability: Liability ):
#     total_assets: float
#     total_liabilities: float
#     new_liability: float
#     income: float
#     monthly_emi: float
#     risk_score: float = 0 

#     newData = [
#         "total_assets": (for e in asset )
#         "total_liabilities": 
#         new_liability: float
#         income: float
#         monthly_emi: float
#         risk_score: float = 0 
#     ]
#     risk = predict_risk(dataset)
#     return risk

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

@app.post("/ai/chat", response_class=PlainTextResponse)
def chat(req: ChatRequest):
    return askAI(req.prompt, req.chatLog, req.asset, req.liability, req.user)

@app.post("/extractPdf")
async def extractPdf(file: UploadFile = File(...)):
    """
    Receives PDF file from Spring Boot,
    extracts bank statements using AI/ML,
    returns list of statements
    """

    # Read file bytes
    pdf_bytes = await file.read()

    statements = readFromPdf(pdf_bytes)

    return statements