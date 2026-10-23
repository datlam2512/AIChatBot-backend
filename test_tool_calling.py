
from langchain_core.messages import HumanMessage

from app.services.tool_graph import tool_graph


def main():
    question = (
    "(125 * 378 + 4500) / 25 - 199"
)

    result = tool_graph.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config={"recursion_limit": 10},
    )

    print("\n===== LỊCH SỬ THỰC THI =====")

    for message in result["messages"]:
        print(
            type(message).__name__,
            ":",
            message.content,
        )

    print("\n===== CÂU TRẢ LỜI CUỐI =====")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
