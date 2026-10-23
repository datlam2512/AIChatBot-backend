
from langchain_core.messages import HumanMessage

from app.services.tool_graph import memory_tool_graph


def chat(question: str, thread_id: str):

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 12,
    }

    result = memory_tool_graph.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config=config,
    )

    print("\nUSER:", question)
    print("AI:", result["messages"][-1].content)


# Cuộc trò chuyện thứ nhất
chat(
    "Tui đang học AI Engineer.",
    "conversation-1"
)

chat(
    "Tui đang học ngành gì?",
    "conversation-1"
)


# Cuộc trò chuyện khác
chat(
    "Tui đang học ReactJS.",
    "conversation-2"
)

chat(
    "Tui đang học gì?",
    "conversation-2"
)


# Kiểm tra dữ liệu Memory
config = {
    "configurable": {
        "thread_id": "conversation-1"
    }
}

state = memory_tool_graph.get_state(config)

print("\nSỐ MESSAGES TRONG THREAD 1:")
print(len(state.values["messages"]))
