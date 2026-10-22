
import numpy as np

from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.utils.smart_chunking import smart_chunk_text
from app.services.embedding_service import generate_embeddings


def save_document(
    db: Session,
    filename: str,
    text: str,
) -> int:

    chunks = smart_chunk_text(
    text,
    chunk_size=500,
    overlap=100,
    )

    if not chunks:
        raise ValueError("Document must not be empty")

    vectors = generate_embeddings(chunks)

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Invalid embeddings shape")

    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain invalid numbers")
    
    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0):
        raise ValueError("Embeddings contain zero vectors")

    document = Document(filename=filename)

    try:
        db.add(document)
        db.flush()

        for chunk, vector in zip(chunks, matrix, strict=True):
            document_chunk = DocumentChunk(
                document_id=document.id,
                content=chunk,
                embedding=vector.tolist(),
            )

            db.add(document_chunk)

        db.commit()

    except Exception:
        db.rollback()
        raise

    return document.id
