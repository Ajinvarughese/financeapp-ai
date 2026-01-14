import os
from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = """
You are a friendly finance assistant.

Reply like WhatsApp messages:
- short and clear
- human tone
- use a few emojis 🙂

You mainly help with finance-related questions (money, savings, loans, budgeting, investing).

You are allowed to:
- remember and repeat basic details the user shares (like their name)
- answer simple memory questions such as "what is my name?"

If the user asks something completely unrelated to finance
(and not about recalling stored user details),
politely reply that you are here to help with financial issues only.
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

def clean_text(text: str) -> str:
    return text.replace("\u0000", "").strip()

def build_messages(user_input, chat_log, asset, liability, user):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.extend(chat_log)

    messages.append({
        "role": "system",
        "content": f"User name: {user}"
    })

    messages.append({
        "role": "system",
        "content": f"Assets: {asset}"
    })

    messages.append({
        "role": "system",
        "content": f"Liabilities: {liability}"
    })

    messages.append({"role": "user", "content": user_input})

    return messages


# ================== AI CALL ==================

def askAI(user_input: str, chat_log: list = [], asset: list = [], liability: list = [], user: str = "") -> str:
    client = get_client()
    messages = build_messages(user_input, chat_log, asset=asset, liability=liability, user=user)

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=4096,
        stream=True
    )

    response_chunks = []

    for chunk in completion:
        if (
            hasattr(chunk, "choices")
            and chunk.choices
            and chunk.choices[0].delta.content
        ):
            response_chunks.append(chunk.choices[0].delta.content)

    response_text = clean_text("".join(response_chunks))

    # update chatLog IN-PLACE
    chat_log.append({"role": "user", "content": clean_text(user_input)})
    chat_log.append({"role": "assistant", "content": response_text})

    return response_text