
from fastapi import APIRouter, HTTPException,Request
from fastapi.responses import StreamingResponse
from app.database import SessionLocal
from app.models import Conversation, Message
from app.schemas.chat import ChatRequest
from sqlalchemy import select
from app.services.chat_prompt_graph import chat_prompt_graph
from app.services.chat_service import stream_and_save
from app.services.message_service import save_message
from app.services.agent_stream_service import (
    stream_agent_and_save,
)

router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)




@router.post("/chat")
def chat(
    request: ChatRequest,
    http_request: Request,
):
    # 1. Kiểm tra request
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty",
        )

    last_message = request.messages[-1]

    if last_message.role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from user",
        )

    # 2. Kiểm tra conversation và lấy lịch sử cũ
    with SessionLocal() as db:

        conversation = db.get(
            Conversation,
            request.conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        old_messages = db.scalars(
            select(Message)
            .where(
                Message.conversation_id
                == request.conversation_id
            )
            .order_by(Message.id.asc())
        ).all()

        bootstrap_history = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in old_messages
            if msg.role in ("user", "assistant")
        ][-20:]

        # Lưu tin nhắn mới như trước
        save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="user",
            content=last_message.content,
        )

    # 3. Lấy Agent có Checkpointer từ FastAPI
    graph = http_request.app.state.agent_graph

    # 4. Stream câu trả lời
    return StreamingResponse(
        stream_agent_and_save(
            graph=graph,
            user_text=last_message.content,
            conversation_id=request.conversation_id,
            bootstrap_history=bootstrap_history,
        ),
        media_type="text/plain; charset=utf-8",
    )
