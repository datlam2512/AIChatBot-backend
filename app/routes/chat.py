
from fastapi import APIRouter, HTTPException,Request,Depends
from fastapi.responses import StreamingResponse
from app.database import SessionLocal
from app.models import Conversation, Message,User
from app.schemas.chat import ChatRequest
from sqlalchemy import select
from app.services.chat_prompt_graph import chat_prompt_graph
from app.services.message_service import save_message
from pydantic import BaseModel
from langgraph.types import Command
from app.services.approval_service import interpret_approval

from app.services.agent_stream_service import (
    stream_agent_and_save,
    stream_simple_reply_and_save,
    resume_approval_and_save,
)
from app.routes.auth import get_current_user

from app.services.conversation_access import (
    get_owned_conversation,
)
router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)
class ApprovalDecision(BaseModel):
    approved: bool


def get_approval_payload(
    graph,
    conversation_id: int,
):
    # 1. Tạo thread_id tương ứng conversation
    config = {
        "configurable": {
            "thread_id": f"conversation-{conversation_id}"
        }
    }

    # 2. Lấy trạng thái đã lưu trong PostgreSQL
    state = graph.get_state(config)

    # 3. Kiểm tra các interrupt đang chờ
    for item in state.interrupts:

        payload = item.value

        if not isinstance(payload, dict):
            continue

        if payload.get("action") == "delete_document":
            return payload

    # 4. Không có yêu cầu xác nhận
    return None



@router.post("/chat")
def chat(
    request: ChatRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    # Kiểm tra tin nhắn
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

    conversation_id = request.conversation_id
    user_text = last_message.content

    # BẮT BUỘC kiểm tra chủ sở hữu
    # trước khi truy cập memory hoặc lưu message
    with SessionLocal() as db:
        get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    graph = http_request.app.state.agent_graph

    # 3. Kiểm tra Agent có đang chờ xác nhận không
    pending = get_approval_payload(
        graph,
        conversation_id,
    )

    if pending is not None:

        if pending.get("action") != "delete_document":
            raise HTTPException(
                status_code=409,
                detail="Unsupported pending action",
            )

        document_id = int(pending["document_id"])
        filename = pending["filename"]

        decision = interpret_approval(
            message=user_text,
            document_id=document_id,
        )

        # Lưu câu user nói vào chat history
        with SessionLocal() as db:
            save_message(
                db=db,
                conversation_id=conversation_id,
                role="user",
                content=user_text,
            )

        # Trường hợp A: Người dùng nói "Ok"
        # hoặc chưa xác nhận đủ rõ
        if decision == "clarify":

            answer = (
                f"Bạn đang xác nhận xóa "
                f"'{filename}' (ID {document_id}) "
                f"đúng không?\n\n"
                f"Nếu đồng ý, hãy nhập "
                f"'Xác nhận xóa {document_id}'. "
                f"Nếu không, hãy nhập 'Hủy'."
            )

            return StreamingResponse(
                stream_simple_reply_and_save(
                    text=answer,
                    conversation_id=conversation_id,
                ),
                media_type="text/plain; charset=utf-8",
            )

        # Trường hợp B: Xác nhận hoặc hủy
        approved = decision == "approve"

        return StreamingResponse(
            resume_approval_and_save(
                graph=graph,
                conversation_id=conversation_id,
                approved=approved,
                document_id=document_id,
                filename=filename,
                user_id=current_user.id,
            ),
            media_type="text/plain; charset=utf-8",
        )


    # 4. Không có pending approval:
    # xử lý chat thông thường như trước
    with SessionLocal() as db:

        old_messages = db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation_id
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

        save_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=user_text,
        )

    return StreamingResponse(
        stream_agent_and_save(
            graph=graph,
            user_text=user_text,
            conversation_id=conversation_id,
            bootstrap_history=bootstrap_history,
            user_id=current_user.id,
        ),
        media_type="text/plain; charset=utf-8",
    )




@router.get("/chat/{conversation_id}/pending-approval")

@router.get("/chat/{conversation_id}/pending-approval")
def get_pending_approval(
    conversation_id: int,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    # Kiểm tra người dùng có sở hữu chat không
    with SessionLocal() as db:
        get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    # Chỉ đọc LangGraph Memory sau khi được phép
    graph = http_request.app.state.agent_graph

    payload = get_approval_payload(
        graph,
        conversation_id,
    )

    if payload is None:
        return {"pending": False}

    return {
        "pending": True,
        **payload,
    }



@router.post("/chat/{conversation_id}/approval")

@router.post("/chat/{conversation_id}/approval")
def respond_to_approval(
    conversation_id: int,
    decision: ApprovalDecision,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    # 1. Xác thực quyền sở hữu conversation
    with SessionLocal() as db:
        get_owned_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    graph = request.app.state.agent_graph

    # 2. Lấy yêu cầu xác nhận đang chờ
    payload = get_approval_payload(
        graph,
        conversation_id,
    )

    if payload is None:
        raise HTTPException(
            status_code=409,
            detail="No pending approval",
        )

    if payload.get("action") != "delete_document":
        raise HTTPException(
            status_code=409,
            detail="Unsupported pending action",
        )

    config = {
    "configurable": {
        "thread_id": f"conversation-{conversation_id}",
        "user_id": current_user.id,
    },
    "recursion_limit": 16,
}

    # 3. Resume đúng tài liệu đang chờ
    result = graph.invoke(
        Command(
            resume={
                "approved": decision.approved,
                "document_id": int(payload["document_id"]),
                "filename": payload["filename"],
            }
        ),
        config=config,
    )

    # 4. Kiểm tra còn interrupt hay không
    if graph.get_state(config).interrupts:
        return {
            "status": "pending",
            "message": "Agent đang chờ xác nhận tiếp.",
        }

    # 5. Lấy câu trả lời cuối
    messages = result.get("messages", [])
    answer = ""

    if messages:
        last_message = messages[-1]
        content = last_message.content

        answer = (
            content
            if isinstance(content, str)
            else str(content)
        )

    # 6. Lưu phản hồi
    if answer:
        with SessionLocal() as db:
            save_message(
                db=db,
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            )

    return {
        "status": (
            "approved" if decision.approved else "cancelled"
        ),
        "message": answer,
    }
