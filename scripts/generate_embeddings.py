import json
from pathlib import Path

from app.core.embeddings import generate_embedding

CHUNKS_PATH = Path("data/processed/chunks.json")
OUTPUT_PATH = Path("data/embeddings/embeddings.json")

OUTPUT_PATH.parent.mkdir(exist_ok=True)

def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings_data = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        embedding = generate_embedding(chunk["text"])

        embeddings_data.append({
            "id": chunk["id"],
            "embedding": embedding,
            "source": chunk["source"],
            "text": chunk["text"]
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f)

    print("Embeddings generation complete")


if __name__ == "__main__":
    main()
