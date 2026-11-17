from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app import NO_DOCUMENT_MESSAGE, DocuMind, to_messages


def make_documind(embeddings, responses, txt_file=None):
    llm = FakeListChatModel(responses=responses)
    dm = DocuMind(embeddings=embeddings, llm=llm)
    if txt_file:
        dm.add_document(txt_file, display_name="contract.txt")
    return dm


def spy_on_search(monkeypatch, dm):
    queries = []
    original = dm.vector_store.similarity_search

    def spy(query, k=4):
        queries.append(query)
        return original(query, k=k)

    monkeypatch.setattr(dm.vector_store, "similarity_search", spy)
    return queries


def test_ask_without_documents(embeddings):
    dm = make_documind(embeddings, ["unused"])
    assert not dm.is_ready()
    assert dm.ask("hello") == {"answer": NO_DOCUMENT_MESSAGE, "sources": []}
    stream, sources = dm.stream_answer("hello")
    assert "".join(stream) == NO_DOCUMENT_MESSAGE and sources == []


def test_ask_returns_answer_and_sources(embeddings, txt_file, monkeypatch):
    dm = make_documind(embeddings, ["30 days notice [contract.txt]"], txt_file)
    queries = spy_on_search(monkeypatch, dm)

    result = dm.ask("What is the notice period?")

    assert result["answer"] == "30 days notice [contract.txt]"
    assert result["sources"] and {s.filename for s in result["sources"]} == {"contract.txt"}
    assert queries == ["What is the notice period?"]  # no history -> no rewrite


def test_follow_up_is_condensed_before_retrieval(embeddings, txt_file, monkeypatch):
    # 1st LLM call: condense step, 2nd: answer
    dm = make_documind(
        embeddings, ["What is the notice period of the termination clause?", "30 days."], txt_file
    )
    queries = spy_on_search(monkeypatch, dm)
    history = [
        {"role": "user", "content": "Is there a termination clause?"},
        {"role": "assistant", "content": "Yes, there is one."},
    ]

    result = dm.ask("And the notice period?", history)

    assert queries == ["What is the notice period of the termination clause?"]
    assert result["answer"] == "30 days."


def test_stream_answer_yields_tokens(embeddings, txt_file):
    dm = make_documind(embeddings, ["Streamed answer"], txt_file)
    stream, sources = dm.stream_answer("Question?")
    chunks = list(stream)
    assert len(chunks) > 1  # really streamed, not one block
    assert "".join(chunks) == "Streamed answer"
    assert sources


def test_clear_knowledge_base(embeddings, txt_file):
    dm = make_documind(embeddings, ["unused"], txt_file)
    assert dm.list_documents() == ["contract.txt"]
    dm.clear_knowledge_base()
    assert dm.list_documents() == [] and not dm.is_ready()


def test_to_messages_keeps_recent_history():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(10)]
    messages = to_messages(history, limit=4)
    assert [m.content for m in messages] == ["6", "7", "8", "9"]
    assert isinstance(messages[0], HumanMessage) and isinstance(messages[1], AIMessage)
    assert to_messages(history, limit=0) == []
