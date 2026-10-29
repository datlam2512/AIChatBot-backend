
import os
from io import BytesIO

import pytest

from dotenv import load_dotenv
from reportlab.pdfgen import canvas

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, delete, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

load_dotenv()

from app.main import app
from app.database import Base, get_db
from app.models import Document, DocumentChunk
from app.services import document_service


# 1. Tạo PDF mẫu ngay trong Python
def make_test_pdf() -> bytes:
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)
    pdf.drawString(
        50, 750,
        "Nova Lab employees have 14 annual leave days."
    )
    pdf.save()

    return buffer.getvalue()


# 2. Giả lập EmbeddingGemma
def fake_embeddings(texts: list[str]):
    return [
        [1.0] + [0.0] * 767
        for _ in texts
    ]


# 3. Chuẩn bị môi trường test
@pytest.fixture
def client(monkeypatch):
    test_url = os.environ["TEST_DATABASE_URL"]
    parsed_url = make_url(test_url)

    # Chặn chạy nhầm trên database chính
    if (
        parsed_url.database != "chatbot_test"
        or parsed_url.port != 5434
    ):
        raise RuntimeError(
            "Refusing to run on a non-test database"
        )

    engine = create_engine(test_url)

    # Bật pgvector trong database test
    with engine.begin() as conn:
        conn.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )

    # Chỉ tạo bảng trong database test
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        with TestSession() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        document_service,
        "generate_embeddings",
        fake_embeddings,
    )

    # Không chạy FastAPI lifespan trong bài test này:
    # chúng ta chỉ test Document API
    test_client = TestClient(app)

    created_ids = []

    try:
        yield test_client, TestSession, created_ids

    finally:
        # Dọn những document do test này tạo ra
        with TestSession() as db:
            for doc_id in created_ids:
                db.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == doc_id
                    )
                )
                db.execute(
                    delete(Document).where(
                        Document.id == doc_id
                    )
                )

            db.commit()

        app.dependency_overrides.pop(get_db, None)
        test_client.close()
        engine.dispose()


# 4. Test cả luồng Upload -> GET -> Duplicate -> Delete
def test_document_lifecycle(client):
    api, TestSession, created_ids = client

    filename = "__pytest_nova_lab.pdf"

    # TEST A: Upload PDF
    response = api.post(
        "/api/documents/upload",
        files={
            "file": (
                filename,
                make_test_pdf(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200, response.text

    document_id = response.json()["document_id"]
    created_ids.append(document_id)

    # Kiểm tra dữ liệu đã lưu thật trong PostgreSQL
    with TestSession() as db:
        chunks = db.scalars(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        ).all()

        assert len(chunks) > 0

        assert any(
            "14 annual leave days" in chunk.content
            for chunk in chunks
        )

    # TEST B: GET ALL
    response = api.get("/api/documents")

    assert response.status_code == 200

    documents = response.json()

    assert any(
        doc["id"] == document_id
        and doc["filename"] == filename
        for doc in documents
    )

    # TEST C: Upload trùng tên
    response = api.post(
        "/api/documents/upload",
        files={
            "file": (
                filename,
                make_test_pdf(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 409

    # TEST D: DELETE
    response = api.delete(
        f"/api/documents/{document_id}"
    )

    assert response.status_code == 200

    # Xác nhận không còn document và chunks
    with TestSession() as db:
        assert db.get(Document, document_id) is None

        remaining = db.scalars(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        ).all()

        assert remaining == []
