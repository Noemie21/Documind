"""Smoke tests of the Streamlit UI with a fake backend (no OpenAI calls)."""
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

from app import DocuMind

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_empty_state_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Upload documents" in info.value for info in at.info)


def test_chat_streams_answer_with_escaped_sources(tmp_path):
    doc = tmp_path / "upload.txt"
    doc.write_text("<script>alert(1)</script> The notice period is 30 days.", encoding="utf-8")
    documind = DocuMind(
        embeddings=DeterministicFakeEmbedding(size=32),
        llm=FakeListChatModel(responses=["30 days [evil.txt]", "Still 30 days."]),
    )
    documind.add_document(str(doc), display_name="evil.txt")

    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["documind"] = documind
    at.session_state["documents"] = ["evil.txt"]
    at.run()

    at.chat_input[0].set_value("What is the notice period?").run()
    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "30 days [evil.txt]"
    assert messages[1]["sources"][0]["filename"] == "evil.txt"

    rendered = " ".join(md.value for md in at.markdown)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


def test_visitor_must_enter_key_when_server_has_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Enter your OpenAI API key" in info.value for info in at.info)
    assert "documind" not in at.session_state  # nothing created without a key

    at.text_input[0].set_value("sk-visitor-key").run()
    assert not at.exception
    assert at.session_state["api_key_in_use"] == "sk-visitor-key"
    assert at.session_state["documind"]._api_key == "sk-visitor-key"
    assert any("Upload documents" in info.value for info in at.info)


def test_no_key_prompt_when_server_has_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-key")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.text_input) == 0
