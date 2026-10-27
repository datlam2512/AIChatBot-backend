
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document


def get_owned_document(
    db: Session,
    document_id: int,
    user_id: int,
) -> Document:

    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return document
