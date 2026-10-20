
from app.database import SessionLocal
from app.services.rag_service import search_relevant_chunks

questions = [
    "Quy định về việc đi muộn tại Nova Lab là gì?",
    "Nhân viên cần cập nhật lý do đi muộn trước mấy giờ?",
    "Nhân viên đi muộn quá 15 phút phải làm gì?"
]

with SessionLocal() as db:
    for question in questions:
        print("\n" + "=" * 60)
        print("QUESTION:", question)

        results = search_relevant_chunks(
            db=db,
            question=question,
            top_k=5
        )

        for i, (chunk, distance) in enumerate(results, start=1):
            print(f"\nTOP {i}")
            print("Distance:", round(float(distance), 4))
            print("Chunk ID:", chunk.id)
            print("Content:", chunk.content[:500])
