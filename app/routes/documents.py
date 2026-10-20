from io import BytesIO

from fastapi import APIRouter, Depends, File , HTTPException, UploadFile
from sqlalchemy.orm import Session
from pypdf import PdfReader

from app.database import get_db
from app.services.document_service import save_document

router = APIRouter(
    prefix = "/api/documents",
    tags = ["Documents"]
)

MAX_FILE_SIZE = 10 * 1024 * 1024 

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
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


    document_id = save_document(
        db= db,
        filename=filename,
        text = text,
    )

    return {
        "message": "PDF processed sucessfully",
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
    }

    