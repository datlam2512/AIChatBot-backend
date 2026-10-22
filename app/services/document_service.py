
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import delete
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


def reindex_document(
    db: Session,
    document_id: int,
    filename: str,
    text: str,
) -> dict:

    # 1. Kiểm tra tài liệu có tồn tại không
    document = db.get(Document, document_id)

    if document is None:
        raise LookupError("Document not found")

    # Tránh cập nhật nhầm PDF khác
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

    # 3. Tạo embeddings mới
    vectors = generate_embeddings(chunks)

    # 4. Kiểm tra embeddings bằng NumPy
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

    # 5. Chỉ bắt đầu thay dữ liệu sau khi
    # chunks và embeddings đã sẵn sàng
    try:
        delete_result = db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        )

        # 6. Lưu chunks mới
        for chunk, vector in zip(
            chunks, matrix, strict=True
        ):
            db.add(
                DocumentChunk(
                    document_id=document_id,
                    content=chunk,
                    embedding=vector.tolist(),
                )
            )

        # 7. Xác nhận toàn bộ thay đổi
        db.commit()

    except Exception:
        # Nếu xảy ra lỗi ghi database,
        # hoàn tác thay đổi trong transaction
        db.rollback()
        raise

    return {
        "document_id": document_id,
        "filename": filename,
        "old_chunks": delete_result.rowcount,
        "new_chunks": len(chunks),
    }
