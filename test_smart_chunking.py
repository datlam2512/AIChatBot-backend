
from pathlib import Path

from pypdf import PdfReader

from app.utils.chunking import chunk_text
from app.utils.smart_chunking import smart_chunk_text


pdf_path = Path("test_files/01_noi_quy_cong_ty.pdf")

reader = PdfReader(str(pdf_path))

text = "\n\n".join(
    page.extract_text() or ""
    for page in reader.pages
)

# Làm sạch Unicode giống API upload
text = text.encode(
    "utf-8", errors="replace"
).decode("utf-8")


# Thuật toán cũ
old_chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=100,
)

# Thuật toán mới
new_chunks = smart_chunk_text(
    text,
    chunk_size=500,
    overlap=100,
)


print("OLD CHUNKS:", len(old_chunks))
print("NEW CHUNKS:", len(new_chunks))


def show_relevant_chunks(chunks, name):
    print(f"\n===== {name} =====")

    for i, chunk in enumerate(chunks, start=1):
        if "15 phút" in chunk or "10:00" in chunk:
            print(f"\nChunk {i}:")
            print(chunk)
            print("-" * 50)


show_relevant_chunks(old_chunks, "OLD CHUNKING")
show_relevant_chunks(new_chunks, "SMART CHUNKING")
