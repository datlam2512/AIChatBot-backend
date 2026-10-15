
from time import perf_counter

from sqlalchemy import select, text

from app.database import SessionLocal
from app.models import DocumentChunk
from app.services.embedding_service import generate_embedding


question = "RAG là gì?"

total_start = perf_counter()

with SessionLocal() as db:

    # Bước 1: Đo kết nối database
    start = perf_counter()

    db.execute(text("SELECT 1"))

    db_time = perf_counter() - start
    print(f"Database connection: {db_time:.3f}s")

    # Bước 2: Đo embedding
    start = perf_counter()

    question_vector = generate_embedding(question)

    embedding_time = perf_counter() - start
    print(f"Embedding: {embedding_time:.3f}s")

    # Bước 3: Đo truy vấn pgvector
    start = perf_counter()

    distance = DocumentChunk.embedding.cosine_distance(
        question_vector
    )

    stmt = (
        select(
            DocumentChunk,
            distance.label("distance")
        )
        .order_by(distance)
        .limit(3)
    )

    results = db.execute(stmt).all()

    search_time = perf_counter() - start
    print(f"Vector search: {search_time:.3f}s")

    # Bước 4: Đo hiển thị
    start = perf_counter()

    for chunk, distance_value in results:
        print(
            f"Distance: {distance_value:.4f} | "
            f"{chunk.content[:100]}"
        )

    print_time = perf_counter() - start
    print(f"Print: {print_time:.3f}s")

total_time = perf_counter() - total_start

print(f"\nTOTAL: {total_time:.3f}s")
