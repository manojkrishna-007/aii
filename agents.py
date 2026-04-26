import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

SYSTEMS = {
    "CEO": "Aggressive CEO. Max 10 words.",
    "CFO": "Risk-averse CFO. Max 10 words.",
    "CTO": "Technical CTO. Max 10 words.",
    "Investor": "ROI-focused investor. Max 10 words.",
    "Director": "You create a business problem in 10 words."
}

def call_llm(role, prompt):
    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": SYSTEMS[role]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return res.choices[0].message.content.strip()