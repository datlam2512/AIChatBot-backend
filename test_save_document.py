
from app.database import SessionLocal
from app.services.document_service import save_document

text = """
Machine Learning là lĩnh vực giúp máy tính học từ dữ liệu
thay vì phải lập trình mọi quy tắc một cách thủ công.

Supervised Learning sử dụng dữ liệu có nhãn để huấn luyện
mô hình dự đoán.

Unsupervised Learning tìm kiếm các cấu trúc hoặc nhóm
trong dữ liệu không có nhãn.

RAG là kỹ thuật kết hợp tìm kiếm thông tin với mô hình
ngôn ngữ lớn để tạo câu trả lời dựa trên tài liệu.
"""

with SessionLocal() as db:
    try:
        document_id = save_document(
            db=db,
            filename="ai_notes.txt",
            text=text,
        )

        print("Document saved successfully!")
        print("Document ID:", document_id)

    except Exception:
        db.rollback()
        raise
