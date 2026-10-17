
import numpy as np

from app.services.embedding_service import (
    generate_embedding,
    generate_embeddings,
)


chunks = [
    "Machine Learning giúp máy tính học từ dữ liệu.",
    "RAG kết hợp tìm kiếm tài liệu và tạo câu trả lời.",
    "Python là ngôn ngữ lập trình.",
    "Nhân viên được nghỉ phép 12 ngày mỗi năm.",
]

question = "RAG hoạt động như thế nào?"
top_k = 3

# 1. Tạo embeddings
chunk_vectors = generate_embeddings(chunks)
question_vector = generate_embedding(question)

# 2. Chuyển thành NumPy arrays
matrix = np.asarray(chunk_vectors, dtype=np.float32)
query = np.asarray(question_vector, dtype=np.float32)

# 3. Kiểm tra số chiều
if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
    raise ValueError("Embedding dimensions do not match")

# 4. Tính Dot Product với tất cả chunks cùng lúc
dot_products = matrix @ query

# 5. Tính norm
chunk_norms = np.linalg.norm(matrix, axis=1)
query_norm = np.linalg.norm(query)

denominators = chunk_norms * query_norm

if np.any(denominators == 0):
    raise ValueError("Cannot compare zero vectors")

# 6. Cosine Similarity
scores = dot_products / denominators

# 7. Lấy chỉ số các chunks có score cao nhất
top_indices = np.argsort(scores)[::-1][:top_k]

# 8. Hiển thị kết quả
for index in top_indices:
    print(f"Score: {scores[index]:.4f}")
    print(f"Chunk: {chunks[index]}")
    print("---")
