
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Message, User
from app.routes.auth import get_current_user

from app.services.conversation_access import (
    get_owned_conversation,
)


router = APIRouter(
    prefix="/api",
    tags=["Messages"],
)


@router.get("/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Kiểm tra quyền sở hữu conversation
    get_owned_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    # Chỉ truy vấn messages sau khi có quyền
    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.id.asc())
        .all()
    )

    return messages
