
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import delete
from app.models import Document, DocumentChunk
from app.utils.smart_chunking import smart_chunk_text
from app.services.embedding_service import generate_embeddings
from sqlalchemy import select, func

class DuplicateDocumentError(Exception):
    pass

def save_document(
    db: Session,
    filename: str,
    text: str,
    user_id: int,
) -> int:
    filename = filename.strip()

    # Kiểm tra file đã tồn tại chưa
    existing = db.scalar(
        select(Document).where(
            Document.filename == filename,
            Document.user_id == user_id,
        )
    )

    if existing is not None:
        raise DuplicateDocumentError(
            f"File '{filename}' đã tồn tại "
            f"(document ID: {existing.id})"
        )
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

    document = Document(filename=filename, user_id=user_id)

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



def reindex_document(
    db: Session,
    document_id: int,
    filename: str,
    text: str,
    user_id: int,
) -> dict:

    # 1. Kiểm tra tài liệu thuộc người dùng
    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
    )

    if document is None:
        raise LookupError("Document not found")

    if document.filename.casefold() != filename.casefold():
        raise ValueError(
            "Filename does not match the existing document"
        )

    # 2. Smart Chunking
    chunks = smart_chunk_text(
        text,
        chunk_size=500,
        overlap=100,
    )

    if not chunks:
        raise ValueError("Document has no text")

    # 3. Generate Embeddings
    vectors = generate_embeddings(chunks)

    matrix = np.asarray(vectors, dtype=np.float32)

    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(chunks)
        or matrix.shape[1] == 0
    ):
        raise ValueError("Invalid embeddings shape")

    if not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding values")

    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0):
        raise ValueError("Zero embedding vector found")

    # 4. Thay thế dữ liệu trong transaction
    try:
        # Khóa và kiểm tra lại quyền trước khi ghi
        document = db.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
            .with_for_update()
        )

        if document is None:
            raise LookupError("Document not found")

        if document.filename.casefold() != filename.casefold():
            raise ValueError(
                "Document filename has changed"
            )

        # Xóa chunks cũ
        delete_result = db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        )

        # Tạo chunks mới
        for chunk, vector in zip(
            chunks,
            matrix,
            strict=True,
        ):
            db.add(
                DocumentChunk(
                    document_id=document_id,
                    content=chunk,
                    embedding=vector.tolist(),
                )
            )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return {
        "document_id": document_id,
        "filename": filename,
        "old_chunks": delete_result.rowcount,
        "new_chunks": len(chunks),
    }

