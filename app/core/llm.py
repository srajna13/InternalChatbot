from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.1-8b-instant"


SYSTEM_PROMPT = """
You are an internal company assistant.
Answer ONLY using the provided context and not in a single word, but full sentences professionally.
If the answer is not present, say:
"I don't have that information in the provided documents."
"""


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)

    return f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You answer using only the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
