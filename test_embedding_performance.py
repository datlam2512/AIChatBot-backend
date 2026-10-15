import time
import httpx

URL = "http://localhost:11434/api/embed"

payload = {
    "model": "embeddinggemma",
    "input": "RAG hoạt động như thế nào?",
    "keep_alive": "30m",
}

for i in range(3):
    start = time.perf_counter()

    response = httpx.post(URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    elapsed = time.perf_counter() - start

    print(f"\nRequest {i + 1}")
    print(f"Client latency: {elapsed:.3f}s")
    print(f"Model load: {data.get('load_duration', 0) / 1e9:.3f}s")
    print(f"Total Ollama: {data.get('total_duration', 0) / 1e9:.3f}s")