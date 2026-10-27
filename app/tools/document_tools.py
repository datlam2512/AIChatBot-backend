
from langchain_core.tools import tool
from sqlalchemy import select, func,delete
from app.database import SessionLocal
from app.models import Document
from app.services.rag_service import search_relevant_chunks
from langgraph.types import interrupt
import ast
import math
import operator
from app.database import SessionLocal
from app.models import Document, DocumentChunk
from langgraph.prebuilt import ToolRuntime

def get_tool_user_id(runtime: ToolRuntime) -> int:
    user_id = runtime.config.get(
        "configurable", {}
    ).get("user_id")

    if type(user_id) is not int or user_id <= 0:
        raise PermissionError(
            "Authenticated user_id is required"
        )

    return user_id


@tool
def get_document_count(runtime: ToolRuntime) -> int:
    """Đếm số tài liệu PDF của người dùng hiện tại."""

    user_id = get_tool_user_id(runtime)

    print("[TOOL] Count documents for user:", user_id)

    with SessionLocal() as db:
        total = db.scalar(
            select(func.count(Document.id))
            .where(Document.user_id == user_id)
        )

    return total or 0




@tool
def list_documents(
    runtime: ToolRuntime,
) -> list[dict]:
    """Liệt kê ID và tên PDF của người dùng hiện tại."""

    user_id = get_tool_user_id(runtime)

    with SessionLocal() as db:
        documents = db.scalars(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.id)
        ).all()

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
            }
            for doc in documents
        ]


@tool
def search_documents(
    question: str,
    runtime: ToolRuntime,
) -> str:
    """
    Tìm kiếm thông tin trong các tài liệu PDF
    thuộc Knowledge Base của người dùng hiện tại.

    Chỉ dùng để tra cứu tài liệu nội bộ, PDF,
    nội quy và các file đã upload.

    Không dùng cho câu hỏi kiến thức chung.
    """

    user_id = get_tool_user_id(runtime)

    print(
        "[TOOL] Search documents:",
        question,
        "User:",
        user_id,
    )

    with SessionLocal() as db:
        results = search_relevant_chunks(
            db=db,
            question=question,
            user_id=user_id,
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

@tool
def request_delete_document(
    document_id: int,
    runtime: ToolRuntime,
) -> str:
    """
    Yêu cầu xóa PDF theo ID.
    Chỉ cho phép xóa tài liệu thuộc người dùng
    hiện tại và sau khi được xác nhận qua HITL.
    """

    user_id = get_tool_user_id(runtime)

    # 1. Kiểm tra tài liệu thuộc người dùng
    with SessionLocal() as db:
        document = db.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )

        if document is None:
            return (
                f"Không tìm thấy tài liệu ID "
                f"{document_id} trong tài khoản của bạn."
            )

        filename = document.filename

    # 2. Yêu cầu xác nhận
    decision = interrupt({
        "action": "delete_document",
        "document_id": document_id,
        "filename": filename,
        "question": (
            f"Bạn có chắc muốn xóa '{filename}'?"
        ),
    })

    # 3. Kiểm tra dữ liệu xác nhận
    if (
        not isinstance(decision, dict)
        or decision.get("approved") is not True
        or decision.get("document_id") != document_id
        or decision.get("filename") != filename
    ):
        return "Đã hủy hoặc từ chối yêu cầu xóa."

    # 4. Kiểm tra quyền lần nữa trước khi DELETE
    with SessionLocal() as db:
        try:
            document = db.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.user_id == user_id,
                )
                .with_for_update()
            )

            if document is None:
                return (
                    "Không tìm thấy tài liệu hoặc "
                    "bạn không còn quyền xóa."
                )

            if document.filename != filename:
                return (
                    "Thông tin tài liệu đã thay đổi. "
                    "Vui lòng thực hiện yêu cầu mới."
                )

            # 5. Xóa chunks và embeddings
            result = db.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id
                    == document_id
                )
            )

            deleted_chunks = result.rowcount

            # 6. Xóa document
            db.delete(document)

            # 7. Commit transaction
            db.commit()

        except Exception:
            db.rollback()
            raise

    return (
        f"Đã xóa thành công '{filename}' "
        f"(ID {document_id}). "
        f"Đã xóa {deleted_chunks} chunks liên quan."
    )


