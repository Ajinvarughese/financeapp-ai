from pydantic import BaseModel

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


class Prompt(BaseModel):
    prompt: str