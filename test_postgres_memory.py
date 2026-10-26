
import os
import sys

from dotenv import load_dotenv
from sqlalchemy.engine import make_url
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from app.services.tool_graph import builder


# 1. Đọc cấu hình database
load_dotenv()

database_url = os.environ["DATABASE_URL"]

# SQLAlchemy thường dùng postgresql+psycopg://
# PostgresSaver cần URL PostgreSQL chuẩn.
db_uri = (
    make_url(database_url)
    .set(drivername="postgresql")
    .render_as_string(hide_password=False)
)

# 2. Sử dụng thread cố định để test
THREAD_ID = "postgres-memory-demo-001"

config = {
    "configurable": {
        "thread_id": THREAD_ID
    },
    "recursion_limit": 12,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"

    # 3. Kết nối PostgreSQL Checkpointer
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:

        # Tạo các bảng checkpoint nếu cần
        checkpointer.setup()

        # 4. Compile graph với PostgreSQL
        graph = builder.compile(
            checkpointer=checkpointer
        )

        if mode == "write":
            question = "Tui đang học AI Engineer."

        elif mode == "read":
            question = "Tui đã nói tui đang học gì?"

        else:
            raise ValueError("Dùng write hoặc read")

        # 5. Xem trạng thái đã lưu trước khi chạy
        previous_state = graph.get_state(config)

        print(
            "[MEMORY] Previous messages:",
            len(previous_state.values.get("messages", []))
        )

        # 6. Chỉ gửi câu hỏi mới
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config,
        )

        print("\nUSER:", question)
        print("AI:", result["messages"][-1].content)

        # 7. Kiểm tra trạng thái sau khi chạy
        state = graph.get_state(config)

        print(
            "\n[MEMORY] Saved messages:",
            len(state.values.get("messages", []))
        )


if __name__ == "__main__":
    main()
