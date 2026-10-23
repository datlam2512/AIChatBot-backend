
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.database import SessionLocal
from app.services.message_service import save_message
from app.services.tool_graph import tool_graph


def stream_agent_and_save(
    messages: list[dict],
    conversation_id: int,
):
    # 1. Chuyển messages từ frontend
    # sang định dạng LangChain
    chat_history = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            chat_history.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            chat_history.append(
                AIMessage(content=content)
            )

    answer_parts = []

    # 2. Chạy LangGraph và nhận text theo từng chunk
    for chunk, metadata in tool_graph.stream(
        {"messages": chat_history},
        stream_mode="messages",
        config={"recursion_limit": 12},
    ):

        # Chỉ lấy output từ node assistant
        if metadata.get("langgraph_node") != "assistant":
            continue

        # Không gửi dữ liệu tool call cho frontend
        if getattr(chunk, "tool_call_chunks", None):
            continue

        # Chỉ gửi text, bỏ qua reasoning và tool metadata
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

        # 3. Stream text tới frontend
        answer_parts.append(text)
        yield text

    # 4. Lưu câu trả lời khi graph chạy xong
    full_answer = "".join(answer_parts)

    if full_answer:
        with SessionLocal() as db:
            save_message(
                db=db,
                conversation_id=conversation_id,
                role="assistant",
                content=full_answer,
            )
