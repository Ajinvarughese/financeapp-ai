from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import List, Dict



class Dataset(BaseModel):
    total_assets: float
    total_liabilities: float
    new_liability: float
    monthly_emi: float
    risk_score: float = 0
    risk_class: str = "SAFE"

    def set_risk_class(self, risk_class: str):
        self.risk_class = risk_class

    def get_risk_class(self):
        return self.risk_class


class ChatRequest(BaseModel):
    prompt: str
    chatLog: List[Dict]
    asset: List[Dict]
    liability: List[Dict]
    user: str



class TransactionType(str, Enum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"

class BankStatement(BaseModel):
    date: datetime          
    particular: str
    refNumber: str
    transactionType: TransactionType
    amount: float

class Asset(BaseModel):
    name: str
    income: float
    expense: float
    debt: float

class Liability(BaseModel):
    name: str
    amount: float
    interest: float
    months: int
    expense: float;

class PredictedResponse(BaseModel):
    riskClass: str
    description: str 