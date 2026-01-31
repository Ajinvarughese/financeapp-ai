import joblib
import numpy as np
from entity import *
import pandas as pd

import tempfile
from datetime import datetime
import camelot
import os


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


def readFromPdf(pdf_bytes: bytes):
    statements = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")

        for table in tables:
            df = table.df

            headers = [h.lower() for h in df.iloc[0].values]
            if "debit" not in headers or "credit" not in headers:
                continue

            df = df.iloc[1:]

            for _, row in df.iterrows():
                try:
                    date = datetime.strptime(row[1].strip(), "%d/%m/%Y")
                except Exception:
                    continue

                particular = row[2].strip()
                ref_number = row[3].strip()
                if not ref_number:
                    continue

                debit = row[4].replace(",", "").strip()
                credit = row[5].replace(",", "").strip()

                if debit and debit != "-" and float(debit) > 0:
                    statements.append(
                        BankStatement(
                            date=date,
                            particular=particular,
                            refNumber=ref_number,
                            transactionType=TransactionType.DEBIT,
                            amount=float(debit)
                        )
                    )

                elif credit and credit != "-" and float(credit) > 0:
                    statements.append(
                        BankStatement(
                            date=date,
                            particular=particular,
                            refNumber=ref_number,
                            transactionType=TransactionType.CREDIT,
                            amount=float(credit)
                        )
                    )

    finally:
        os.remove(pdf_path)

    return statements
