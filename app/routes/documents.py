from io import BytesIO

from fastapi import APIRouter, Depends, File , HTTPException, UploadFile
from sqlalchemy.orm import Session
from pypdf import PdfReader
from sqlalchemy import delete
from app.database import get_db
from app.services.document_service import (
    save_document,
    reindex_document,
)
from sqlalchemy import select, func
from app.models import Document, DocumentChunk,User
from app.services.document_service import DuplicateDocumentError
from app.routes.auth import get_current_user
from app.services.document_access import get_owned_document
router = APIRouter(
    prefix = "/api/documents",
    tags = ["Documents"]
)

MAX_FILE_SIZE = 10 * 1024 * 1024 



@router.get("")
def get_all_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(
            Document.id,
            Document.filename,
            Document.create_at,
            func.count(DocumentChunk.id).label("chunk_count"),
        )
        .outerjoin(
            DocumentChunk,
            DocumentChunk.document_id == Document.id,
        )
        .where(
            Document.user_id == current_user.id
        )
        .group_by(
            Document.id,
            Document.filename,
            Document.create_at,
        )
        .order_by(Document.id.desc())
    )

    results = db.execute(stmt).all()

    return [
        {
            "id": row.id,
            "filename": row.filename,
            "created_at": row.create_at,
            "chunk_count": row.chunk_count,
        }
        for row in results
    ]




@router.get("/{document_id}")
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = get_owned_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id,
    )

    chunk_count = db.scalar(
        select(func.count(DocumentChunk.id))
        .where(
            DocumentChunk.document_id == document.id
        )
    )

    return {
        "id": document.id,
        "filename": document.filename,
        "created_at": document.create_at,
        "chunk_count": chunk_count,
    }


@router.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code= 400,
            detail = "Only PDF files are supported"
        )

    data = file.file.read(MAX_FILE_SIZE + 1)

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds 10 MB"
        )

    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code= 400,
            detail = "Invalid PDF file"
        )

    try:
        reader = PdfReader(BytesIO(data))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDF is not supported")

        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]

    except Exception as exc:
        raise HTTPException(
            status_code= 400,
            detail = "Cannot read this PDF"
        ) from exc


    text = "\n\n".join(pages)

# Loại bỏ các ký tự Unicode không hợp lệ
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    if not text.strip():
        raise HTTPException(
           status_code=400,
           detail="No extractable text found in PDF"
    )


    
    try:
        document_id = save_document(
        db=db,
        filename=filename,
        text=text,
        user_id=current_user.id,
)

    except DuplicateDocumentError as exc:
        raise HTTPException(
        status_code=409,
        detail=str(exc),
    ) from exc


    return {
        "message": "PDF processed sucessfully",
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
    }

    
@router.post("/{document_id}/reindex")
def reindex_pdf(
    document_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    
    filename = file.filename or ""
    
    # 1. Kiểm tra file PDF
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    data = file.file.read(MAX_FILE_SIZE + 1)

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds 10 MB",
        )

    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file",
        )

    # 2. Đọc nội dung PDF
    try:
        reader = PdfReader(BytesIO(data))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDF")

        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Cannot read this PDF",
        ) from exc

    text = "\n\n".join(pages)

    # Xử lý lỗi Unicode đã gặp trước đây
    text = text.encode(
        "utf-8", errors="replace"
    ).decode("utf-8")

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found",
        )

    # 3. Re-index
    try:
        result = reindex_document(
              db=db,
              document_id=document_id,
              filename=filename,
              text=text,
              user_id=current_user.id,
)

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "message": "Re-index successful",
        **result,
    }

@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Kiểm tra chủ sở hữu và khóa bản ghi
    document = db.scalar(
        select(Document)
        .where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
        .with_for_update()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    filename = document.filename

    try:
        # 2. Xóa chunks và embeddings
        result = db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document.id
            )
        )

        deleted_chunks = result.rowcount

        # 3. Xóa document
        db.delete(document)
        db.commit()

    except Exception:
        db.rollback()
        raise

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "filename": filename,
        "deleted_chunks": deleted_chunks,
    }
