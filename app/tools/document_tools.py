
from langchain_core.tools import tool
from sqlalchemy import select, func

from app.database import SessionLocal
from app.models import Document
from app.services.rag_service import search_relevant_chunks

import ast
import math
import operator

from langchain_core.tools import tool
@tool
def get_document_count() -> int:
    """Lấy tổng số tài liệu PDF hiện có trong database."""

    print("[TOOL] Đang đếm tài liệu...")

    with SessionLocal() as db:
        total = db.scalar(
            select(func.count(Document.id))
        )

    return total or 0


@tool
def list_documents() -> list[dict]:
    """Lấy danh sách tên và ID của các tài liệu PDF trong hệ thống."""

    print("[TOOL] Đang lấy danh sách tài liệu...")

    with SessionLocal() as db:
        documents = db.scalars(
            select(Document).order_by(Document.id)
        ).all()

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
            }
            for doc in documents
        ]
@tool
def search_documents(question: str) -> str:
    """
    Tìm kiếm thông tin trong các tài liệu PDF
    do người dùng upload vào Knowledge Base.

    Chỉ sử dụng khi câu hỏi yêu cầu tra cứu
    tài liệu nội bộ, quy định, nội dung PDF
    hoặc thông tin từ các file đã upload.

    KHÔNG sử dụng cho câu hỏi kiến thức chung
    như "Python là gì?" hoặc "AI là gì?".
    """

    print("[TOOL] Đang tìm trong tài liệu:", question)

    with SessionLocal() as db:
        results = search_relevant_chunks(
            db=db,
            question=question,
            top_k=5,
        )

    if not results:
        return "Không tìm thấy nội dung tài liệu."

    context = "\n\n".join(
        f"[Đoạn {i}]\n{chunk.content}"
        for i, (chunk, distance) in enumerate(
            results,
            start=1,
        )
    )

    return context

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def evaluate(node):
    # Một con số: 10, 3.5...
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value

    # Phép toán: +, -, *, /...
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in OPERATORS:
            raise ValueError("Phép toán không hỗ trợ")

        left = evaluate(node.left)
        right = evaluate(node.right)

        if op_type is ast.Pow and abs(right) > 10:
            raise ValueError("Số mũ quá lớn")

        return check_result(
            OPERATORS[op_type](left, right)
        )

    # Số âm hoặc số dương: -5, +3
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value

    # Hàm căn bậc hai: sqrt(81)
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return math.sqrt(evaluate(node.args[0]))

    raise ValueError("Biểu thức không được hỗ trợ")


def check_result(value):
    if not math.isfinite(value) or abs(value) > 1e15:
        raise ValueError("Kết quả vượt giới hạn")
    return value


@tool
def calculator(expression: str) -> str:
    """
    Tính biểu thức toán học gồm:
    cộng, trừ, nhân, chia, chia lấy dư,
    lũy thừa, ngoặc và căn bậc hai.

    Ví dụ: (14 - 5) * 2, sqrt(81), 2 ** 10.
    """

    print("[TOOL] Calculating:", expression)

    try:
        if len(expression) > 100:
            return "Lỗi: Biểu thức quá dài"

        tree = ast.parse(expression, mode="eval")

        result = check_result(evaluate(tree.body))

        return str(result)

    except (ValueError, ZeroDivisionError, SyntaxError,
            OverflowError, TypeError, RecursionError):
        return "Không thể tính biểu thức này"