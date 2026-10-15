
from app.database import SessionLocal
from app.services.rag_service import (
    search_relevant_chunks,
    build_rag_messages,
)
from app.services.llm_service import generate_response


question = "Machine Learning là gì?"

messages = [
    {"role": "user", "content": question}
]

with SessionLocal() as db:
    # Bước 1: Tìm tài liệu
    results = search_relevant_chunks(
        db=db,
        question=question,
        top_k=3,
    )

    # Bước 2: Lấy nội dung chunks
    chunks = [chunk.content for chunk, distance in results]

    # Bước 3: Tạo messages chứa context
    rag_messages = build_rag_messages(
        messages=messages,
        chunks=chunks,
    )

    # Bước 4: Kiểm tra prompt
    print("=== RAG PROMPT ===")
    print(rag_messages[0]["content"])

    # Bước 5: Gọi Qwen
    print("\n=== QWEN RESPONSE ===")

    for token in generate_response(rag_messages):
        print(token, end="", flush=True)

    print()
