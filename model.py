import joblib
import numpy as np
from entity import *
import pandas as pd

import tempfile
from datetime import datetime
import camelot
from chat_bot import recommendation


def saveDataset(total_assets, total_liabilities, new_liability, income, monthly_emi, risk_score):
    new_row = {
        "Total Assets": total_assets,
        "Total Liabilities": total_liabilities,
        "New Liability Amount": new_liability,
        "Income": income,
        "Monthly EMI": monthly_emi,
        "Risk Score": risk_score
    }

    df = pd.DataFrame([new_row])
    df.to_csv("dataset/financial_risk_dataset.csv", mode="a", header=False, index=False)




# Load model and encoder
model = joblib.load("dataset/risk_model.pkl")
label_encoder = joblib.load("dataset/risk_label_encoder.pkl")

def predict_risk(dataset: Dataset) -> PredictedResponse:
    """
    Predict risk of a new liability.
    Business rules first, then ML for borderline cases.
    """

    total_assets = dataset.total_assets
    total_liabilities = dataset.total_liabilities
    new_liability = dataset.new_liability
    monthly_emi = dataset.monthly_emi

    # ---------- 1. Business Rule Layer ----------
    financial_score = total_assets - total_liabilities

    base = financial_score if financial_score > 0 else 1
    risk_ratio = new_liability / base

    if financial_score <= 0:
        description = recommendation("NOT_RECOMMENDED", dataset)
        return PredictedResponse(
            riskClass="NOT_RECOMMENDED",
            description=description
        )

    if risk_ratio > 0.60:
        description = recommendation("NOT_RECOMMENDED", dataset)
        return PredictedResponse(
            riskClass="NOT_RECOMMENDED",
            description=description
        )

    if 0.25 < risk_ratio <= 0.60:
        description = recommendation("RISKY", dataset)
        return PredictedResponse(
            riskClass="RISKY",
            description=description
        )

    # ---------- 2. ML Model Layer ----------
    asset_buffer = total_assets - total_liabilities

    debt_ratio = total_liabilities / total_assets if total_assets > 0 else 0
    new_liability_ratio = new_liability / asset_buffer if asset_buffer > 0 else 0
    emi_ratio = monthly_emi / total_assets if total_assets > 0 else 0

    arr = np.array([[
        total_assets,
        total_liabilities,
        new_liability,
        monthly_emi,
        asset_buffer,
        debt_ratio,
        new_liability_ratio,
        emi_ratio
    ]])

    pred_encoded = model.predict(arr)[0]
    riskClass = label_encoder.inverse_transform([pred_encoded])[0]
    riskClass = riskClass.upper().replace(" ", "_")

    description = recommendation(riskClass, dataset)

    return PredictedResponse(
        riskClass=riskClass,
        description=description
    )



def readFromPdf(pdf_bytes: bytes):
    statements = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    tables = camelot.read_pdf(pdf_path, pages="1", flavor="lattice")

    for table in tables:
        df = table.df

        if "Debit" not in df.iloc[0].values:
            continue

        df = df.iloc[1:]

        for _, row in df.iterrows():
            date = datetime.strptime(row[1].strip(), "%d/%m/%Y")
            particular = row[2].strip()

            debit = row[4].replace(",", "").strip()
            credit = row[5].replace(",", "").strip()

            if debit and debit != "-":
                statements.append(
                    BankStatement(
                        date=date,
                        particular=particular,
                        transactionType=TransactionType.DEBIT,
                        amount=float(debit)
                    )
                )
            elif credit and credit != "-":
                statements.append(
                    BankStatement(
                        date=date,
                        particular=particular,
                        transactionType=TransactionType.CREDIT,
                        amount=float(credit)
                    )
                )

    return statements
