
from langchain_text_splitters import RecursiveCharacterTextSplitter


def smart_chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap")

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,

        separators=[
            "\n\n",  # Ưu tiên tách đoạn văn
            "\n",    # Sau đó tách dòng
            ". ",    # Kết thúc câu
            "! ",
            "? ",
            "; ",
            " ",     # Tách từ
            "",      # Cuối cùng mới cắt ký tự
        ],

        keep_separator="end",
    )

    return splitter.split_text(text)
