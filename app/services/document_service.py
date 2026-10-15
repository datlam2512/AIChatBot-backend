from sqlalchemy.orm import Session
from app.models import Document, DocumentChunk
from app.utils.chunking import chunk_text
from app.services.embedding_service import generate_embeddings

def save_document(db:Session, filename: str, text: str) -> int:
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    if not chunks:
        raise ValueError("Document must not be empty")
    vectors = generate_embeddings(chunks)
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")
    document = Document(filename=filename)
    db.add(document)
    db.flush()
    for chunk, vector in zip(chunks, vectors, strict=True):
        document_chunk = DocumentChunk(
            document_id = document.id,
            content = chunk,
            embedding = vector,
        )
        db.add(document_chunk)
    db.commit()
    return document.id