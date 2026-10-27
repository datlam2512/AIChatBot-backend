import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends,HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Conversation, User,Message
from app.routes.auth import get_current_user
from sqlalchemy import delete
from app.services.conversation_access import (
    get_owned_conversation,
)
router = APIRouter(
    prefix="/api",
    tags=["Conversations"]
)


class ConversationCreate(BaseModel):
    title: str


# ======================================
# GET CONVERSATIONS
# ======================================

@router.get("/conversations")
def get_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id
        )
        .all()
    )

    return conversations


# ======================================
# CREATE CONVERSATION
# ======================================

@router.post("/conversations", status_code=201)
def create_conversation(
    data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.now(timezone.utc)

    conversation = Conversation(
        title=data.title,
        user_id=current_user.id,
        created_at=now,
        updated_at=now,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation

logger = logging.getLogger(__name__)


@router.delete(
    "/conversations/{conversation_id}"
)
def delete_conversation(
    conversation_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Kiểm tra conversation thuộc user
    conversation = get_owned_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    # 2. Xóa messages và conversation trong
    # cùng một transaction
    try:
        db.execute(
            delete(Message).where(
                Message.conversation_id
                == conversation_id
            )
        )

        db.delete(conversation)

        db.commit()

    except Exception:
        db.rollback()
        raise

    # 3. Xóa AI Memory trong LangGraph
    graph = request.app.state.agent_graph

    thread_id = f"conversation-{conversation_id}"

    checkpoint_deleted = False

    try:
        checkpointer = graph.checkpointer

        checkpointer.delete_thread(thread_id)

        checkpoint_deleted = True

    except Exception:
        logger.exception(
            "Failed to clean checkpoints for %s",
            thread_id,
        )

    return {
        "message": "Conversation deleted",
        "conversation_id": conversation_id,
        "checkpoint_deleted": checkpoint_deleted,
    }
