import requests
import chromadb
from chromadb.config import Settings
import textwrap

DB_PATH = "./chroma_db"
COLLECTION_NAME = "obsidian_notes"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "zeno_wizard"

# -----------------------------
# Embedding helper
# -----------------------------

def embed_query(text):
    """Embed a user query using Ollama's HTTP API."""
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    )
    data = response.json()

    if "embeddings" not in data:
        raise ValueError(f"Embedding API returned unexpected response: {data}")

    return data["embeddings"][0]

# -----------------------------
# LLM helper
# -----------------------------

def ask_llm(question, context):
    """Send question + retrieved context to Zeno_Wizard"""
    prompt = f"""
You are a research assistant with access to the user's personal notes.

Use ONLY the context below to answer the question. 
If the context is insufficient, say so explicitly.

CONTEXT:
{textwrap.indent(context, "  ")}

QUESTION:
{question}

Answer clearly and concisely.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt}
    )

    # Streamed responses come in chunks; concatenate them
    answer = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            answer += data

    return answer

# -----------------------------
# Main interactive loop
# -----------------------------

def main():
    print("RAG system ready. Ask questions about your Obsidian notes.")

    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings())
    collection = client.get_collection(COLLECTION_NAME)

    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        # Embed the question
        q_emb = embed_query(q)

        # Retrieve top chunks
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=5
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Build context block
        context_parts = []
        for doc, meta in zip(docs, metas):
            context_parts.append(
                f"[{meta['path']} chunk {meta['chunk']}]\n{doc}\n"
            )
        context = "\n".join(context_parts)

        # Ask WizardLM-2
        answer = ask_llm(q, context)

        print("\n=== ANSWER ===\n")
        print(answer)

if __name__ == "__main__":
    main()
