from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.database import SessionLocal
from app.models import Conversation
from app.schemas.chat import ChatRequest
from app.services.chat_service import stream_and_save
from app.services.message_service import save_message

router = APIRouter(prefix="/api", tags=["Chat"])

@router.post("/chat")
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code = 400, detail="Last message must be from the user")
    with SessionLocal() as db:
        conversation = db.get(Conversation, request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code = 404, detail = "Conversation not found")
        save_message(
            db=db,
            conversation_id = request.conversation_id,
            role = "user",
            content = last_message.content,
        )
    messages = [
        message.model_dump()
        for message in request.messages
    ]
    return StreamingResponse(
        stream_and_save(messages, request.conversation_id),
        media_type = "text/plain"
    )