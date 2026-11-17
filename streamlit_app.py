"""
DocuMind - Streamlit Frontend
Chat interface for document Q&A
"""

import html
import os
import tempfile

import streamlit as st

from app import DocuMind

# Page config
st.set_page_config(
    page_title="DocuMind - AI Document Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.85em;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (each browser session gets its own in-memory collection)
if "documind" not in st.session_state:
    st.session_state.documind = DocuMind()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = st.session_state.documind.list_documents()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def render_sources(sources):
    """Render source chunks; all document-derived text is HTML-escaped."""
    with st.expander("📚 Sources"):
        for src in sources:
            filename = html.escape(str(src["filename"]))
            content = html.escape(str(src["content"]))
            page = f" - Page {src['page']}" if src.get("page") else ""
            st.markdown(f"""
            <div class="source-box">
                <strong>{filename}</strong>{page}<br>
                <em>{content}</em>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Header
    st.title("🧠 DocuMind")
    st.markdown("*Chat with your documents using AI*")
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.documents:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save to temp file
                        suffix = os.path.splitext(uploaded_file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            # Add to knowledge base
                            chunks = st.session_state.documind.add_document(
                                tmp_path, display_name=uploaded_file.name
                            )
                            st.session_state.documents.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name} ({chunks} chunks)")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
        
        # Show loaded documents
        if st.session_state.documents:
            st.markdown("---")
            st.subheader("Loaded documents")
            for doc in st.session_state.documents:
                st.markdown(f"• {doc}")
            st.caption("Documents are kept for this browser session only.")

            if st.session_state.messages and st.button("💬 New conversation"):
                st.session_state.messages = []
                st.rerun()

            if st.button("🗑️ Clear all documents", type="secondary"):
                st.session_state.documind.clear_knowledge_base()
                st.session_state.documents = []
                st.session_state.messages = []
                # New uploader key, otherwise the files still in the widget get re-indexed
                st.session_state.uploader_key += 1
                st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("""
        ### How to use
        1. Upload one or more documents
        2. Ask questions in the chat
        3. Get answers with source citations
        
        ### Features
        - 📄 Multi-document support
        - 🔍 Semantic search
        - 📝 Source citations
        - 💬 Follow-up questions (conversation memory)
        - ⚡ Streaming responses
        """)
    
    # Main chat area
    if not st.session_state.documents:
        st.info("👈 Upload documents to get started")
        
        # Demo section
        st.markdown("---")
        st.markdown("### 🎯 What can DocuMind do?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **📚 Research**
            - Analyze academic papers
            - Extract key findings
            - Compare methodologies
            """)
        with col2:
            st.markdown("""
            **💼 Business**
            - Query contracts
            - Summarize reports
            - Find specific clauses
            """)
        with col3:
            st.markdown("""
            **📖 Learning**
            - Study materials
            - Quick fact lookup
            - Concept explanations
            """)
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # History *before* this question, used to understand follow-ups
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate a streamed response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching your documents..."):
                        stream, sources = st.session_state.documind.stream_answer(prompt, history)
                    answer = st.write_stream(stream)
                except Exception as e:
                    st.error(f"Error while generating the answer: {e}")
                    st.session_state.messages.pop()  # drop the unanswered question
                    return

                sources = [s.to_dict() for s in sources]
                if sources:
                    render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })


if __name__ == "__main__":
    main()
