def chunk_text(text: str, chunk_size: int=500, overlap: int=100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must bbe between 0 and chunk_size")

    chunks = []
    start=0

    while start < len(text):
        end = min(start + chunk_size , len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks