import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from entity import Dataset
from entity import PredictedResponse

# ================== SYSTEM PROMPTS ==================

SYSTEM_PROMPT = """
You are a friendly finance assistant.

Reply like WhatsApp messages:
- short and clear
- human tone
- use a few emojis 🙂

You mainly help with finance-related questions (money, savings, loans, budgeting, investing).

If the user asks something completely unrelated to finance,
politely reply that you are here to help with financial issues only.
"""


SYSTEM_PROMPT_FOR_RISK_CLASSIFICATION = """
You are a financial risk classification engine.

Classify whether taking a new liability is SAFE, RISKY, or NOT_RECOMMENDED.

Rules:
- Output ONLY one word.
- Valid outputs are: SAFE, RISKY, NOT_RECOMMENDED.
- Do NOT explain.
- Do NOT include punctuation or extra text.

Decision criteria:
- Monthly income
- Monthly expenses
- New liability EMI
- Remaining disposable income after EMI
- Overall affordability of the EMI

Guidelines:
- SAFE: EMI is comfortably affordable and disposable income remains high.
- RISKY: EMI is affordable but significantly reduces financial buffer.
- NOT_RECOMMENDED: EMI causes financial stress or leaves insufficient disposable income.

Affordability thresholds:
- SAFE if EMI ≤ 40% of monthly income
- RISKY if EMI is between 40% and 80% of monthly income
- NOT_RECOMMENDED if EMI > 80% of monthly income

"""

SYSTEM_PROMPT_FOR_DESCRIPTION = """
You are a financial summary generator.

Rules:
- Write exactly ONE paragraph.
- 2–3 complete sentences only.
- Do NOT end mid-sentence.
- Ensure the last sentence is fully completed.
- No emojis.
- No bullet points.
- No line breaks.
- No greetings.
- Professional, clear tone.
"""

# ================== CLIENT (cached) ==================

_client = None

def get_client():
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found")
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    return _client

# ================== HELPERS ==================

def clean_text(text) -> str:
    if not text or not isinstance(text, str):
        return ""
    return text.replace("\u0000", "").strip()


def build_messages(user_input, chat_log, asset, liability, user):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_log)

    messages.append({"role": "system", "content": f"User name: {user}"})
    messages.append({"role": "system", "content": f"Assets: {asset}"})
    messages.append({"role": "system", "content": f"Liabilities: {liability}"})
    messages.append({"role": "user", "content": user_input})

    return messages

# ================== AI CHAT ==================

def askAI(
    user_input: str,
    chat_log: list = [],
    asset: list = [],
    liability: list = [],
    user: str = ""
) -> str:
    client = get_client()
    messages = build_messages(user_input, chat_log, asset, liability, user)

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=1,
        max_tokens=4096,
        stream=True
    )

    response_chunks = []

    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            response_chunks.append(chunk.choices[0].delta.content)

    response_text = clean_text("".join(response_chunks))

    chat_log.append({"role": "user", "content": clean_text(user_input)})
    chat_log.append({"role": "assistant", "content": response_text})

    return response_text

# ================== RISK CLASSIFICATION ==================


def safe_message_content(msg) -> str:
    if not msg:
        return ""
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, list):
        return " ".join(
            part.get("text", "") for part in msg if isinstance(part, dict)
        ).strip()
    return ""

def aiRiskPrediction(dataset: Dataset, prompt: str, risk_class: str ="") -> str:
    client = get_client()

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"""
            Total monthly income: {dataset.total_assets}rs
            Total monthly expense: {dataset.total_liabilities}rs
            Monthly EMI of new Liability: {dataset.monthly_emi}rs
            """
        }
    ]
    if risk_class != "":
        messages.append({
            "role": "user",
            "content": f"""
            Predicted risk class: {risk_class}
            """
        })

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0,
        max_tokens=4096
    )

    return safe_message_content(completion.choices[0].message.content)

def aiAnalysis(dataset: Dataset) -> str:
    risk = aiRiskPrediction(dataset=dataset, prompt=SYSTEM_PROMPT_FOR_RISK_CLASSIFICATION)
    return risk

def aiRecommendation(dataset: Dataset, risk_class: str) -> str:
    description = aiRiskPrediction(dataset=dataset,prompt=SYSTEM_PROMPT_FOR_DESCRIPTION, risk_class=risk_class)
    return description  
