import faiss
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# ---------- SAMPLE DOCUMENTS ----------
documents = [
    "The refund policy was updated in 2022.",
    "Users can request refunds within 30 days of purchase.",
    "Premium subscriptions include priority support."
]

# ---------- EMBEDDING MODEL ----------
embedder = SentenceTransformer(EMBED_MODEL)
doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]

# ---------- FAISS INDEX ----------
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ---------- RETRIEVE ----------
def retrieve(query, k=TOP_K):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    return [documents[i] for i in indices[0]]

# ---------- GENERATE (OLLAMA) ----------
def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

# ---------- GROUNDEDNESS ----------
def compute_groundedness(answer, context_chunks):
    sentences = re.split(r'(?<=[.!?]) +', answer)
    context_embeddings = embedder.encode(context_chunks)

    scores = []

    for sentence in sentences:
        s_emb = embedder.encode([sentence])[0]
        similarities = np.dot(context_embeddings, s_emb) / (
            np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(s_emb)
        )
        scores.append(max(similarities))

    groundedness = float(np.mean(scores))
    return groundedness, scores

# ---------- RUN ----------
if __name__ == "__main__":
    question = "When was the refund policy updated?"

    retrieved = retrieve(question)
    answer = generate_answer(question, retrieved)
    groundedness, sentence_scores = compute_groundedness(answer, retrieved)

    print("\nQuestion:", question)
    print("\nRetrieved Context:", retrieved)
    print("\nAnswer:", answer)
    print("\nGroundedness Score:", groundedness)
    print("\nSentence Scores:", sentence_scores)
