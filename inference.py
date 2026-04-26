import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

AGENTS = ["CEO","CFO","CTO","Investor"]

print("\nBOARDROOM AI (10-WORD EXECUTIVE MODE)\n")

TASK = "Global expansion under risk and budget constraints"

for r in range(3):

    print(f"\nROUND {r+1}\n")

    outputs = []

    for a in AGENTS:

        res = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role":"user","content":f"{a} respond in 10 words: {TASK}"}]
        )

        msg = " ".join(res.choices[0].message.content.split()[:10])

        print(f"{a}: {msg}")

        outputs.append(msg)

    if len(set(outputs)) <= 2:
        break

print("\nFINAL DECISION: PHASED EXPANSION")