
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.database import SessionLocal
from app.models import Conversation
from app.schemas.chat import ChatRequest

from app.services.chat_service import stream_and_save
from app.services.message_service import save_message
from app.services.rag_service import (
    build_rag_messages,
    search_relevant_chunks,
)


router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat")
def chat(request: ChatRequest):

    # 1. Kiểm tra messages
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="No messages provided."
        )

    last_message = request.messages[-1]

    if last_message.role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from the user"
        )

    # 2. Mở database session
    with SessionLocal() as db:

        # Kiểm tra conversation
        conversation = db.get(
            Conversation,
            request.conversation_id
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

        # 3. Tìm các chunks liên quan
        results = search_relevant_chunks(
            db=db,
            question=last_message.content,
            top_k=3,
        )

        # 4. Lấy nội dung chunks
        chunks = [
            chunk.content
            for chunk, distance in results
        ]

        # 5. Lưu câu hỏi thật của user
        save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="user",
            content=last_message.content,
        )

    # 6. Chuyển messages thành list[dict]
    messages = [
        msg.model_dump()
        for msg in request.messages
    ]

    # 7. Tạo RAG prompt
    rag_messages = build_rag_messages(
        messages=messages,
        chunks=chunks,
    )

    # 8. Streaming response từ Qwen
    return StreamingResponse(
        stream_and_save(
            rag_messages,
            request.conversation_id,
        ),
        media_type="text/plain",
    )
