from pathlib import Path

from app.core.embeddings import generate_embedding
from app.vectorstore.faiss_store import FAISSVectorStore, load_embeddings
# from app.core.rag_pipeline import build_prompt
from app.core.llm import call_llm,build_prompt


EMBEDDINGS_PATH = Path("data/embeddings/embeddings.json")


def main():
    embeddings, metadata = load_embeddings(EMBEDDINGS_PATH)

    store = FAISSVectorStore(dim=len(embeddings[0]))
    store.add(embeddings, metadata)

    question = "Who is our founder and ceo?"

    query_embedding = generate_embedding(question)
    results = store.search(query_embedding, top_k=3)

    chunks = [r["text"] for r in results]

    prompt = build_prompt(question, chunks)
    answer = call_llm(prompt)

    print("\nANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
