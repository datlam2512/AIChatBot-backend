import math
from app.services.embedding_service import (generate_embeddings,generate_embedding)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_roduct = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a == 0  or norm_b == 0:
        return 0.0
    return dot_roduct/(norm_a*norm_b)

chunks = [
    "Nhân viên chính thức được nghỉ phép 12 ngày mỗi năm.",
    "Công ty cung cấp bảo hiểm sức khỏe cho nhân viên.",
    "Python là ngôn ngữ lập trình thường được sử dụng trong AI.",
]

question = "Một năm tôi được nghỉ bao nhiêu ngày?"

chunk_vectors = generate_embeddings(chunks)
question_vectors = generate_embedding(question)

results = []

for chunk, vector in zip(chunks, chunk_vectors):
    score = cosine_similarity(question_vectors, vector)
    results.append((chunk, score))

results.sort(key=lambda item: item[1], reverse= True)

for chunk, score in results:
    print(f"Score: {score:.4f} | {chunk}")

print("\nChunk liên quan nhất:")
print(results[0][0])