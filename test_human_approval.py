
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict
from sqlalchemy.engine import make_url

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver


class ApprovalState(TypedDict, total=False):
    document_id: int
    status: str


def approval_node(state: ApprovalState):

    # 1. Dừng lại và chờ người dùng xác nhận
    approved = interrupt({
        "action": "delete_document",
        "document_id": state["document_id"],
        "question": "Bạn có chắc muốn xóa PDF này?",
    })

    # 2. Nếu từ chối thì hủy thao tác
    if approved is not True:
        return {"status": "cancelled"}

    # 3. Chỉ mô phỏng xóa, chưa đụng database
    print(
        "[DEMO] Would delete document:",
        state["document_id"]
    )

    return {"status": "approved_demo"}


# Xây graph
builder = StateGraph(ApprovalState)

builder.add_node("approval", approval_node)

builder.add_edge(START, "approval")
builder.add_edge("approval", END)


# PostgreSQL đang dùng trong project
url = make_url(os.environ["DATABASE_URL"])

db_uri = (
    url.set(drivername="postgresql")
    .render_as_string(hide_password=False)
)

THREAD_ID = "hitl-demo-001"

config = {
    "configurable": {"thread_id": THREAD_ID}
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "request"

    with PostgresSaver.from_conn_string(db_uri) as saver:
        saver.setup()

        graph = builder.compile(checkpointer=saver)

        if mode == "request":
            result = graph.invoke(
                {"document_id": 3},
                config=config,
            )

            print("INTERRUPT:", result.get("__interrupt__"))

        elif mode in ("approve", "reject"):
            result = graph.invoke(
                Command(resume=(mode == "approve")),
                config=config,
            )

            print("STATUS:", result["status"])

        else:
            raise ValueError("Use request, approve or reject")


if __name__ == "__main__":
    main()
