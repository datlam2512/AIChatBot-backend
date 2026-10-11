from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models import Message

def save_message(db: Session, conversation_id: int, role:str, content:str):
    message = Message(
        conversation_id = conversation_id,
        role = role,
        content = content,
        created_at = datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message