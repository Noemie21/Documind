from langchain_core.documents import Document

from app import VectorStore


def doc(text, filename):
    return Document(page_content=text, metadata={"filename": filename})


def test_sessions_are_isolated(embeddings):
    a = VectorStore(embeddings=embeddings)
    b = VectorStore(embeddings=embeddings)
    a.add_documents([doc("alpha", "a.txt")])
    b.add_documents([doc("beta", "b.txt")])

    assert a.list_filenames() == ["a.txt"]
    assert b.list_filenames() == ["b.txt"]
    assert {d.metadata["filename"] for d in a.similarity_search("beta", k=4)} == {"a.txt"}


def test_clear_only_affects_own_collection(embeddings):
    a = VectorStore(embeddings=embeddings)
    b = VectorStore(embeddings=embeddings)
    a.add_documents([doc("alpha", "a.txt")])
    b.add_documents([doc("beta", "b.txt")])

    a.clear()
    assert a.list_filenames() == []
    assert b.list_filenames() == ["b.txt"]

    a.add_documents([doc("again", "c.txt")])  # usable after clear
    assert a.list_filenames() == ["c.txt"]


def test_persisted_collection_is_reloaded(tmp_path, embeddings):
    directory = str(tmp_path / "db")
    first = VectorStore(persist_directory=directory, collection_name="kb-test", embeddings=embeddings)
    first.add_documents([doc("kept", "kept.txt")])

    reopened = VectorStore(persist_directory=directory, collection_name="kb-test", embeddings=embeddings)
    assert reopened.load_existing()
    assert reopened.list_filenames() == ["kept.txt"]


def test_empty_store_does_not_load(tmp_path, embeddings):
    store = VectorStore(persist_directory=str(tmp_path / "missing"), embeddings=embeddings)
    assert not store.load_existing()
