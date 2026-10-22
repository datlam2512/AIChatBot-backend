
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.database import SessionLocal
from app.models import Conversation
from app.schemas.chat import ChatRequest

from app.services.chat_prompt_graph import chat_prompt_graph
from app.services.chat_service import stream_and_save
from app.services.message_service import save_message


router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)


@router.post("/chat")
def chat(request: ChatRequest):

    # 1. Kiểm tra request
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty"
        )

    last_message = request.messages[-1]

    if last_message.role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from user"
        )

    # 2. Kiểm tra conversation
    with SessionLocal() as db:
        conversation = db.get(
            Conversation,
            request.conversation_id
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

    # 3. Chuyển Pydantic messages thành dictionaries
    messages = [
        msg.model_dump()
        for msg in request.messages
    ]

    # 4. Chạy LangGraph để chọn prompt
    result = chat_prompt_graph.invoke({
        "question": last_message.content,
        "messages": messages,
    })

    llm_messages = result["llm_messages"]

    print("[CHAT] Selected mode:", result["mode"])

    # 5. Lưu tin nhắn user
    with SessionLocal() as db:
        save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="user",
            content=last_message.content
        )

    # 6. Qwen streaming + lưu câu trả lời assistant
    return StreamingResponse(
        stream_and_save(
            llm_messages,
            request.conversation_id
        ),
        media_type="text/plain"
    )
