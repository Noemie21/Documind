import os

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

# No test talks to OpenAI: embeddings and LLM are always faked.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def txt_file(tmp_path):
    path = tmp_path / "tmpab12cd.txt"
    path.write_text(
        "DocuMind answers questions about documents. " * 60
        + "\n\nThe termination clause requires 30 days notice.",
        encoding="utf-8",
    )
    return str(path)
