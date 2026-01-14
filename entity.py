from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import List, Dict

class Dataset(BaseModel):
    total_assets: float
    total_liabilities: float
    new_liability: float
    income: float
    monthly_emi: float
    risk_score: float = 0 

    def set_risk_score(self, risk_score: float):
        self.risk_score = risk_score

    def get_risk_score(self):
        return self.risk_score


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
    transactionType: TransactionType
    amount: float

class Asset(BaseModel):
    source: str
    income: float
    expense: float
    debt: float

class Liability(BaseModel):
    name: str
    amount: float
    interest: float
    months: int
    expense: float;

