"""
DocuMind - RAG Backend
A simple but production-ready RAG pipeline for document Q&A
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Source:
    """Represents a source document chunk"""
    content: str
    page: Optional[int]  # 1-based page number, None for formats without pages
    filename: str


class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Any]:
        """Load a document based on its extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
        }
        
        loader_class = loaders.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {ext}")
        
        loader = loader_class(file_path)
        return loader.load()
    
    def process_document(self, file_path: str, display_name: Optional[str] = None) -> List[Any]:
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
        
        chunks = self.text_splitter.split_documents(documents)
        return chunks


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
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name or f"documind-{uuid.uuid4().hex}"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None

    def create_from_documents(self, documents: List[Any]) -> None:
        """Create a new vector store from documents"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents: List[Any]) -> None:
        """Add documents to existing vector store"""
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
    
    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, k: int = 4):
        """Get retriever interface"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def clear(self) -> None:
        """Delete this store's collection (other collections are untouched)"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
        self.vectorstore = None


class RAGChain:
    """Main RAG chain for question answering"""
    
    def __init__(self, vector_store: VectorStore, model: str = "gpt-4o-mini"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model, temperature=0, streaming=True)
        self.chain = None
        self._setup_chain()
    
    def _setup_chain(self) -> None:
        """Setup the RAG chain"""
        template = """You are a helpful assistant that answers questions based on the provided context.

RULES:
- Answer ONLY based on the context provided
- If the context doesn't contain enough information, say "I don't have enough information to answer this question."
- Always cite your sources by mentioning the document name and page number
- Be concise but complete

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                filename = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page")
                source = f"[{filename} - Page {page}]" if page else f"[{filename}]"
                formatted.append(f"{source}\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)
        
        self.format_docs = format_docs
        
        retriever = self.vector_store.as_retriever(k=4)
        
        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """Ask a question and get an answer"""
        if self.chain is None:
            raise ValueError("Chain not initialized")
        return self.chain.invoke(question)
    
    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        docs = self.vector_store.similarity_search(question, k=4)
        context = self.format_docs(docs)
        
        template = """You are a helpful assistant that answers questions based on the provided context.

RULES:
- Answer ONLY based on the context provided
- If the context doesn't contain enough information, say "I don't have enough information to answer this question."
- Always cite your sources by mentioning the document name and page number
- Be concise but complete

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({"context": context, "question": question})
        
        sources = [
            Source(
                content=doc.page_content[:200] + "...",
                page=doc.metadata.get("page"),
                filename=doc.metadata.get("filename", "Unknown")
            )
            for doc in docs
        ]
        
        return {"answer": answer, "sources": sources}
    
    def stream(self, question: str):
        """Stream the response"""
        if self.chain is None:
            raise ValueError("Chain not initialized")
        return self.chain.stream(question)


class DocuMind:
    """Main application class"""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self.rag_chain = None

        if self.vector_store.load_existing():
            self.rag_chain = RAGChain(self.vector_store)

    def add_document(self, file_path: str, display_name: Optional[str] = None) -> int:
        """Add a document to the knowledge base"""
        chunks = self.processor.process_document(file_path, display_name=display_name)
        self.vector_store.add_documents(chunks)
        self.rag_chain = RAGChain(self.vector_store)
        return len(chunks)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question"""
        if self.rag_chain is None:
            return {"answer": "Please upload a document first.", "sources": []}
        return self.rag_chain.query_with_sources(question)
    
    def stream_answer(self, question: str):
        """Stream an answer"""
        if self.rag_chain is None:
            yield "Please upload a document first."
            return
        for chunk in self.rag_chain.stream(question):
            yield chunk
    
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
