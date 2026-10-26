
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from app.database import SessionLocal
from app.services.message_service import save_message


def stream_agent_and_save(
    graph,
    user_text: str,
    conversation_id: int,
    bootstrap_history: list[dict],
):
    # 1. Một conversation tương ứng một thread
    config = {
        "configurable": {
            "thread_id": f"conversation-{conversation_id}"
        },
        "recursion_limit": 16,
    }

    # 2. Xem thread đã có Memory chưa
    snapshot = graph.get_state(config)

    if snapshot.values.get("messages"):

        # Đã có checkpoint:
        # chỉ gửi câu hỏi mới
        input_messages = [
            HumanMessage(content=user_text)
        ]

        print("[MEMORY] Restored existing thread")

    else:

        # Thread chưa có checkpoint:
        # nạp lịch sử cũ từ bảng messages một lần
        input_messages = []

        for message in bootstrap_history:

            if message["role"] == "user":
                input_messages.append(
                    HumanMessage(content=message["content"])
                )

            elif message["role"] == "assistant":
                input_messages.append(
                    AIMessage(content=message["content"])
                )

        # Thêm câu hỏi hiện tại
        input_messages.append(
            HumanMessage(content=user_text)
        )

        print("[MEMORY] Initialized thread from history")

    # 3. Chạy Agent với persistent checkpoint
    answer_parts = []

    for chunk, metadata in graph.stream(
        {"messages": input_messages},
        config=config,
        stream_mode="messages",
    ):

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
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                )
            )

        else:
            text = ""

        if not text:
            continue

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
