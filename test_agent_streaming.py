
from langchain_core.messages import HumanMessage

from app.services.tool_graph import tool_graph


question = "Hệ thống hiện có bao nhiêu PDF?"

print("QUESTION:", question)
print("\nANSWER:")

for chunk, metadata in tool_graph.stream(
    {
        "messages": [
            HumanMessage(content=question)
        ]
    },
    stream_mode="messages",
    config={"recursion_limit": 12},
):
    if metadata.get("langgraph_node") != "assistant":
        continue

    if getattr(chunk, "tool_call_chunks", None):
        continue

    if isinstance(chunk.content, str):
        print(chunk.content, end="", flush=True)

print("\n")
