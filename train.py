import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("dataset/financial_risk_dataset.csv")


X = df[["Total Assets", "Total Liabilities", "New Liability Amount", "Income", "Monthly EMI"]]
y = df["Risk Score"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)


# 5. Split into Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6. Create and Train Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)

# 8. Save trained model + label encoder
joblib.dump(model, "dataset/risk_model.pkl")
joblib.dump(label_encoder, "dataset/risk_label_encoder.pkl")