
from app.services.chat_graph import chat_graph


while True:
    question = input("\nBạn: ").strip()

    if question.lower() in ("exit", "quit"):
        break

    if not question:
        continue

    result = chat_graph.invoke({
        "question": question
    })

    print("\nAI:", result["answer"])
