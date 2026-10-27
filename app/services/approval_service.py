
import re
import unicodedata
from typing import Literal


ApprovalAction = Literal[
    "approve",
    "reject",
    "clarify",
]


def normalize_message(text: str) -> str:
    # Chuyển về chữ thường và bỏ dấu tiếng Việt
    text = unicodedata.normalize(
        "NFD", text.casefold().strip()
    )

    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.replace("đ", "d")

    # Bỏ dấu câu đơn giản và khoảng trắng thừa
    text = re.sub(r"[.!?,]+", " ", text)
    text = " ".join(text.split())

    return text


def interpret_approval(
    message: str,
    document_id: int,
) -> ApprovalAction:

    text = normalize_message(message)

    # Người dùng muốn hủy
    if text in {
        "huy",
        "huy bo",
        "khong",
        "khong xoa",
        "dung xoa",
        "thoi",
        "cancel",
    }:
        return "reject"

    # Chỉ chấp nhận câu xác nhận rõ hành động và ID
    confirmations = {
        f"xac nhan xoa",
        f"xac nhan xoa tai lieu {document_id}",
        f"xac nhan xoa file {document_id}",
        f"dong y xoa tai lieu {document_id}",
    }

    if text in confirmations:
        return "approve"

    # "Ok", "ừ", "đồng ý", hoặc câu chưa rõ
    # đều cần xác nhận cụ thể hơn
    return "clarify"
