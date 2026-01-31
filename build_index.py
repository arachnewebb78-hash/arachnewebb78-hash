import os
import requests
import chromadb
from chromadb.config import Settings

# -----------------------------
# CONFIGURATION
# -----------------------------

VAULT_PATH = "/Users/timothylake/Obsidian/Second_Brain"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "obsidian_notes"
EMBED_MODEL = "nomic-embed-text"

# -----------------------------
# HELPERS
# -----------------------------

def chunk_text(text, max_chars=800, overlap=200):
    """Split long markdown text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def embed_text(text):
    try:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": text}
        )
        data = response.json()

        # Ollama 0.15.x returns "embeddings": [[...]]
        if "embeddings" in data:
            return data["embeddings"][0]

        print("Embedding API returned no embeddings:", data)
        return None

    except Exception as e:
        print(f"Embedding failed: {e}")
        return None


def embed_texts(texts):
    """Embed a list of text chunks."""
    embeddings = []
    for t in texts:
        emb = embed_text(t)
        if emb is not None:
            embeddings.append(emb)
        else:
            embeddings.append([0.0] * 768)  # fallback vector
    return embeddings


# -----------------------------
# MAIN INDEXING LOGIC
# -----------------------------

def main():
    print("Starting Obsidian → Chroma indexer...")
    print(f"Vault: {VAULT_PATH}")

    # Ensure DB directory exists
    os.makedirs(DB_PATH, exist_ok=True)

    # Initialize Chroma
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True))
    client.reset()
    collection = client.create_collection(COLLECTION_NAME)

    docs = []
    metadatas = []
    ids = []

    # Walk the vault
    for root, _, files in os.walk(VAULT_PATH):
        for fname in files:
            if not fname.endswith(".md"):
                continue

            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                doc_id = f"{fpath}-{i}"
                docs.append(chunk)
                metadatas.append({"path": fpath, "chunk": i})
                ids.append(doc_id)

    print(f"Collected {len(docs)} text chunks.")
    print("Embedding...")

    embeddings = embed_texts(docs)

    print("Adding to Chroma...")
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    print("Indexing complete.")
    print(f"Database stored at: {DB_PATH}")


if __name__ == "__main__":
    main()
