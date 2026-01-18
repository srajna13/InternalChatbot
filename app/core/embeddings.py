from openai import OpenAI;
# import os;
# from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embedding(text):
    embedding= model.encode(text, normalize_embeddings=True)
    return embedding.tolist()