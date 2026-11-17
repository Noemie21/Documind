"""
DocuMind - RAG Backend
RAG pipeline for document Q&A with conversation memory and streaming.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Configuration (overridable through environment variables) ---------------
CHUNK_SIZE = int(os.getenv("DOCUMIND_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("DOCUMIND_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("DOCUMIND_TOP_K", "4"))
LLM_MODEL = os.getenv("DOCUMIND_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("DOCUMIND_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_HISTORY_MESSAGES = int(os.getenv("DOCUMIND_MAX_HISTORY", "6"))

NO_DOCUMENT_MESSAGE = "Please upload a document first."

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

RULES:
- Answer ONLY based on the context provided
- If the context doesn't contain enough information, say "I don't have enough information to answer this question."
- Always cite your sources by mentioning the document name and page number (when available)
- Be concise but complete

CONTEXT:
{context}"""

CONDENSE_SYSTEM_PROMPT = """Given the conversation so far and a follow-up question, rewrite the \
follow-up question as a standalone question that can be understood without the conversation. \
Do NOT answer it. If it is already standalone, return it unchanged. Reply with the question only."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "Follow-up question: {question}"),
])


@dataclass
class Source:
    """Represents a source document chunk"""
    content: str
    page: Optional[int]  # 1-based page number, None for formats without pages
    filename: str

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "page": self.page, "filename": self.filename}


ChatHistory = Sequence[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]


def to_messages(history: Optional[ChatHistory], limit: int = MAX_HISTORY_MESSAGES) -> List[BaseMessage]:
    """Convert UI chat history into LangChain messages, keeping the last `limit`."""
    if limit <= 0:
        return []
    messages: List[BaseMessage] = []
    for msg in list(history or [])[-limit:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks as context for the LLM"""
    formatted = []
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page")
        label = f"[{filename} - Page {page}]" if page else f"[{filename}]"
        formatted.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def to_sources(docs: List[Document], preview_chars: int = 200) -> List[Source]:
    """Build short source previews from retrieved chunks"""
    sources = []
    for doc in docs:
        content = doc.page_content
        if len(content) > preview_chars:
            content = content[:preview_chars] + "..."
        sources.append(Source(
            content=content,
            page=doc.metadata.get("page"),
            filename=doc.metadata.get("filename", "Unknown"),
        ))
    return sources


class DocumentProcessor:
    """Handles document loading and chunking"""

    LOADERS = {
        ".pdf": PyPDFLoader,
        ".txt": lambda path: TextLoader(path, autodetect_encoding=True),
        ".docx": Docx2txtLoader,
    }

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its extension"""
        ext = os.path.splitext(file_path)[1].lower()
        loader_factory = self.LOADERS.get(ext)
        if not loader_factory:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_factory(file_path).load()

    def process_document(self, file_path: str, display_name: Optional[str] = None) -> List[Document]:
        """Load and chunk a document.

        `display_name` is the name shown in citations (e.g. the original upload
        name when `file_path` points to a temporary file).
        """
        documents = self.load_document(file_path)

        # Add filename metadata and normalize page numbers (1-based, PDF only)
        filename = display_name or os.path.basename(file_path)
        for doc in documents:
            doc.metadata["filename"] = filename
            page = doc.metadata.get("page")
            if isinstance(page, int):
                doc.metadata["page"] = page + 1  # PyPDFLoader is 0-based
            else:
                doc.metadata.pop("page", None)

        return self.text_splitter.split_documents(documents)


class VectorStore:
    """Manages the vector database.

    Each instance owns its own Chroma collection, so two sessions never see
    each other's documents. By default the store is in-memory; pass a
    `persist_directory` (and a stable `collection_name`) to keep it on disk.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name or f"documind-{uuid.uuid4().hex}"
        if embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.embeddings = embeddings
        self.vectorstore: Optional[Chroma] = None

    def create_from_documents(self, documents: List[Document]) -> None:
        """Create a new vector store from documents"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store"""
        if not documents:
            return
        if self.vectorstore is None:
            self.create_from_documents(documents)
        else:
            self.vectorstore.add_documents(documents)

    def load_existing(self) -> bool:
        """Load an existing persisted collection, if it has documents"""
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            return False
        store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        if not store.get(limit=1)["ids"]:
            return False
        self.vectorstore = store
        return True

    def list_filenames(self) -> List[str]:
        """Names of the documents currently stored"""
        if self.vectorstore is None:
            return []
        metadatas = self.vectorstore.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("filename", "Unknown") for m in metadatas if m})

    def similarity_search(self, query: str, k: int = TOP_K) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.similarity_search(query, k=k)

    def clear(self) -> None:
        """Delete this store's collection (other collections are untouched)"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
        self.vectorstore = None


class RAGChain:
    """RAG question answering with conversation memory.

    Follow-up questions are first rewritten into standalone questions using the
    chat history, so retrieval works for questions like "and what about the
    second point?". The answer step also sees the recent history.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model: str = LLM_MODEL,
        llm: Optional[BaseChatModel] = None,
        k: int = TOP_K,
    ):
        self.vector_store = vector_store
        self.k = k
        if llm is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=0, streaming=True)
        self.llm = llm
        self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()

    def condense_question(self, question: str, history: List[BaseMessage]) -> str:
        """Rewrite a follow-up question into a standalone one (no-op without history)"""
        if not history:
            return question
        standalone = self.condense_chain.invoke({"history": history, "question": question}).strip()
        return standalone or question

    def _prepare(self, question: str, history: Optional[ChatHistory]) -> Tuple[Dict[str, Any], List[Document]]:
        messages = to_messages(history)
        search_query = self.condense_question(question, messages)
        docs = self.vector_store.similarity_search(search_query, k=self.k)
        inputs = {"context": format_docs(docs), "history": messages, "question": question}
        return inputs, docs

    def query_with_sources(self, question: str, history: Optional[ChatHistory] = None) -> Dict[str, Any]:
        """Ask a question and get the full answer with its sources"""
        inputs, docs = self._prepare(question, history)
        answer = self.answer_chain.invoke(inputs)
        return {"answer": answer, "sources": to_sources(docs)}

    def stream_with_sources(
        self, question: str, history: Optional[ChatHistory] = None
    ) -> Tuple[Iterator[str], List[Source]]:
        """Retrieve sources, then return (token stream, sources)"""
        inputs, docs = self._prepare(question, history)
        return self.answer_chain.stream(inputs), to_sources(docs)

    def query(self, question: str, history: Optional[ChatHistory] = None) -> str:
        """Ask a question and get an answer"""
        return self.query_with_sources(question, history)["answer"]


class DocuMind:
    """Main application class"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        self._llm = llm
        self.rag_chain: Optional[RAGChain] = None

        if self.vector_store.load_existing():
            self.rag_chain = RAGChain(self.vector_store, llm=self._llm)

    def add_document(self, file_path: str, display_name: Optional[str] = None) -> int:
        """Add a document to the knowledge base, returns the number of chunks"""
        chunks = self.processor.process_document(file_path, display_name=display_name)
        self.vector_store.add_documents(chunks)
        if chunks and self.rag_chain is None:
            self.rag_chain = RAGChain(self.vector_store, llm=self._llm)
        return len(chunks)

    def ask(self, question: str, history: Optional[ChatHistory] = None) -> Dict[str, Any]:
        """Ask a question, returns {"answer": str, "sources": [Source]}"""
        if self.rag_chain is None:
            return {"answer": NO_DOCUMENT_MESSAGE, "sources": []}
        return self.rag_chain.query_with_sources(question, history)

    def stream_answer(
        self, question: str, history: Optional[ChatHistory] = None
    ) -> Tuple[Iterator[str], List[Source]]:
        """Stream an answer, returns (token stream, sources)"""
        if self.rag_chain is None:
            return iter([NO_DOCUMENT_MESSAGE]), []
        return self.rag_chain.stream_with_sources(question, history)

    def clear_knowledge_base(self) -> None:
        """Clear all documents"""
        self.vector_store.clear()
        self.rag_chain = None

    def list_documents(self) -> List[str]:
        """Names of the documents in the knowledge base"""
        return self.vector_store.list_filenames()

    def is_ready(self) -> bool:
        """Check if the system is ready"""
        return self.rag_chain is not None
