
import pytest

from app.utils.smart_chunking import smart_chunk_text


def test_empty_text():
    result = smart_chunk_text("")
    assert result == []


def test_short_text():
    text = "Python là một ngôn ngữ lập trình."

    result = smart_chunk_text(text)

    assert result == [text]


def test_chunk_size_limit():
    text = "Python là ngôn ngữ lập trình. " * 100

    chunks = smart_chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        smart_chunk_text(
            "Hello",
            chunk_size=0,
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        smart_chunk_text(
            "Hello",
            chunk_size=100,
            overlap=100,
        )
