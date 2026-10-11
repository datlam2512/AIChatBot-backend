from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database  import get_db
from app.models import Conversation
from pydantic import BaseModel
from datetime import datetime, timezone
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Conversations"])

class ConversationCreate(BaseModel): 
    title: str

@router.get("/conversations")
def get_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversation).all()
    return conversations

@router.post("/conversations", status_code= 201)
def create_conversation(data: ConversationCreate, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    conversation = Conversation(
        title = data.title,
        created_at = now,
        updated_at = now
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation