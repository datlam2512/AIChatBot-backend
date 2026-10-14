import httpx
import os
from dotenv import load_dotenv
load_dotenv()

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = httpx.post(
        OLLAMA_EMBED_URL, 
        json={ 
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout= 120.0,
        )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"]

def generate_embedding(text:str) -> list[float]:
    if not text.strip():
        raise ValueError("Text must not be empty")
    embeddings = generate_embeddings([text])
    return embeddings[0]