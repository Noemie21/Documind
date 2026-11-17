import pytest
from langchain_core.documents import Document

from app import DocumentProcessor, format_docs, to_sources


def test_display_name_replaces_temp_file_name(txt_file):
    chunks = DocumentProcessor().process_document(txt_file, display_name="contract.txt")
    assert len(chunks) > 1
    assert {c.metadata["filename"] for c in chunks} == {"contract.txt"}


def test_txt_chunks_have_no_page(txt_file):
    chunks = DocumentProcessor().process_document(txt_file)
    assert all("page" not in c.metadata for c in chunks)


def test_pdf_pages_are_one_based(monkeypatch):
    processor = DocumentProcessor()
    monkeypatch.setattr(
        processor,
        "load_document",
        lambda path: [
            Document(page_content="first", metadata={"page": 0}),
            Document(page_content="second", metadata={"page": 1}),
        ],
    )
    chunks = processor.process_document("report.pdf")
    assert [c.metadata["page"] for c in chunks] == [1, 2]


def test_non_utf8_text_is_loaded(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("Résumé du contrat : préavis de trente jours.".encode("latin-1"))
    chunks = DocumentProcessor().process_document(str(path))
    assert "préavis" in chunks[0].page_content


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentProcessor().load_document("notes.xlsx")


def test_format_docs_labels():
    docs = [
        Document(page_content="a", metadata={"filename": "r.pdf", "page": 3}),
        Document(page_content="b", metadata={"filename": "n.txt"}),
    ]
    context = format_docs(docs)
    assert "[r.pdf - Page 3]" in context
    assert "[n.txt]" in context and "Page None" not in context


def test_to_sources_truncates_long_content():
    docs = [
        Document(page_content="x" * 500, metadata={"filename": "a.txt"}),
        Document(page_content="short", metadata={"filename": "b.txt", "page": 2}),
    ]
    long_src, short_src = to_sources(docs)
    assert long_src.content == "x" * 200 + "..."
    assert short_src.content == "short" and short_src.page == 2
