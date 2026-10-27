from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from app.database import SessionLocal
from app.services.message_service import save_message
from langgraph.types import Command


def stream_agent_and_save(
    graph,
    user_text: str,
    conversation_id: int,
    bootstrap_history: list[dict],
    user_id: int,
):
    if type(user_id) is not int or user_id <= 0:
        raise PermissionError("Invalid authenticated user")

    config = {
        "configurable": {
            "thread_id": f"conversation-{conversation_id}",
            "user_id": user_id,
        },
        "recursion_limit": 16,
    }

    # 2. Xem thread đã có Memory chưa
    snapshot = graph.get_state(config)

    if snapshot.values.get("messages"):
        # Đã có checkpoint:
        # chỉ gửi câu hỏi mới
        input_messages = [HumanMessage(content=user_text)]

        print("[MEMORY] Restored existing thread")

    else:
        # Thread chưa có checkpoint:
        # nạp lịch sử cũ từ bảng messages một lần
        input_messages = []

        for message in bootstrap_history:
            if message["role"] == "user":
                input_messages.append(HumanMessage(content=message["content"]))

            elif message["role"] == "assistant":
                input_messages.append(AIMessage(content=message["content"]))

        # Thêm câu hỏi hiện tại
        input_messages.append(HumanMessage(content=user_text))

        print("[MEMORY] Initialized thread from history")

    # 3. Chạy Agent với persistent checkpoint
    answer_parts = []

    for chunk, metadata in graph.stream(
        {"messages": input_messages},
        config=config,
        stream_mode="messages",
    ):
        if graph.get_state(config).interrupts and not answer_parts:
            text = "Mình cần bạn xác nhận thao tác ở bên dưới."
            answer_parts.append(text)
            yield text
        # Chỉ lấy các token từ Assistant
        if metadata.get("langgraph_node") != "assistant":
            continue

        # Không hiển thị nội dung tool call
        if getattr(chunk, "tool_call_chunks", None):
            continue

        content = chunk.content

        if isinstance(content, str):
            text = content

        elif isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if (isinstance(block, dict) and block.get("type") == "text")
            )

        else:
            text = ""

        if not text:
            continue

        answer_parts.append(text)
        yield text

    # Kiểm tra Agent có vừa tạo interrupt không
    state = graph.get_state(config)

    if state.interrupts:
        payload = state.interrupts[0].value

        if isinstance(payload, dict) and payload.get("action") == "delete_document":
            document_id = payload["document_id"]
            filename = payload["filename"]

            text = (
                f"\n\nBạn có chắc muốn xóa "
                f"'{filename}' (ID {document_id})?\n\n"
                f"Nhập 'Xác nhận xóa' "
                f"hoặc 'Hủy'."
            )

            answer_parts.append(text)
            yield text

    # 4. Giữ hệ thống lưu lịch sử cho frontend
    full_answer = "".join(answer_parts)

    if full_answer:
        with SessionLocal() as db:
            save_message(
                db=db,
                conversation_id=conversation_id,
                role="assistant",
                content=full_answer,
            )


def stream_simple_reply_and_save(
    text: str,
    conversation_id: int,
):
    # Dùng cho những câu cần xác nhận rõ hơn
    yield text

    with SessionLocal() as db:
        save_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=text,
        )


def resume_approval_and_save(
    graph,
    conversation_id: int,
    approved: bool,
    document_id: int,
    filename: str,
    user_id: int,
):
    if type(user_id) is not int or user_id <= 0:
        raise PermissionError("Invalid authenticated user")

    config = {
        "configurable": {
        "thread_id": f"conversation-{conversation_id}",
        "user_id": user_id,
    },
    "recursion_limit": 16,
}
    # Tiếp tục đúng graph đang bị interrupt

    result = graph.invoke(
        Command(
            resume={
                "approved": approved,
                "document_id": document_id,
                "filename": filename,
            }
        ),
        config=config,
    )

    # Kiểm tra còn yêu cầu xác nhận khác không
    state = graph.get_state(config)

    if state.interrupts:
        answer = (
            "Agent đang chờ một xác nhận khác. Vui lòng kiểm tra yêu cầu tiếp theo."
        )
    else:
        answer = ""

        # Lấy câu trả lời cuối cùng của Assistant
        last_message = result.get("messages", [])[-1:]

        if last_message:
            message = last_message[0]

            if (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and isinstance(message.content, str)
            ):
                answer = message.content.strip()

        if not answer:
            answer = "Agent đã xử lý quyết định, nhưng chưa tạo được câu trả lời cuối."

    # Giữ cùng giao thức text/plain với frontend
    yield answer

    # Tiếp tục lưu lịch sử cho Next.js
    with SessionLocal() as db:
        save_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )
