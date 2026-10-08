from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database  import get_db
from app.models import Conversation

router = APIRouter()

@router.get("/api/conversations")
def get_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversation).all()
    return conversations