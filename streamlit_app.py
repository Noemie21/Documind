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

# Custom CSS on top of the theme in .streamlit/config.toml (matches the noemie.pro portfolio).
# Selectors use Streamlit's data-testid attributes, which are the most stable hooks available.
st.markdown("""
<style>
    :root {
        --dm-violet: #8b5cf6;
        --dm-violet-light: #a78bfa;
        --dm-card: #18181b;
        --dm-border: #27272a;
        --dm-muted: #94a3b8;
        --dm-gradient: linear-gradient(90deg, #a78bfa, #c084fc, #f472b6);
    }

    /* Layout */
    .block-container { max-width: 1000px; padding-top: 2.5rem; }
    [data-testid="stHeader"] { background: transparent; }

    /* Hero */
    .dm-hero h1 {
        font-size: clamp(2.2rem, 5vw, 3rem); font-weight: 700; letter-spacing: -0.03em;
        margin: 0; padding: 0; color: #fff;
    }
    .dm-hero .gradient-text {
        background: var(--dm-gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .dm-hero p { color: var(--dm-muted); font-size: 1.05rem; margin: 0.35rem 0 1.5rem; }
    .dm-eyebrow {
        font-family: 'JetBrains Mono', monospace; color: var(--dm-violet-light);
        font-size: 0.85rem; margin-bottom: 0.4rem;
    }

    /* Feature cards (empty state) */
    .dm-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
    .dm-card {
        background: var(--dm-card); border: 1px solid var(--dm-border); border-radius: 1rem;
        padding: 1.25rem 1.25rem 1rem; transition: transform .25s ease, box-shadow .25s ease;
    }
    .dm-card:hover { transform: translateY(-4px); box-shadow: 0 20px 40px -15px rgba(139, 92, 246, .3); }
    .dm-card h4 { margin: 0 0 .5rem; font-size: 1.05rem; color: #fff; }
    .dm-card ul { margin: 0; padding-left: 1.1rem; color: var(--dm-muted); font-size: .92rem; }
    .dm-tags { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1.5rem 0 0; }
    .dm-tag {
        background: rgba(139, 92, 246, .1); color: var(--dm-violet-light);
        padding: .2rem .75rem; border-radius: 999px; font-size: .85rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--dm-card); border: 1px solid var(--dm-border); border-radius: 1rem;
        padding: 1rem 1.1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(139, 92, 246, .1); border-color: rgba(139, 92, 246, .35);
    }
    [data-testid="stChatMessageAvatarAssistant"] { background: var(--dm-gradient); }
    [data-testid="stChatMessageAvatarUser"] { background: #7c3aed; }

    /* Chat input */
    [data-testid="stChatInput"] { border-radius: 1rem; }
    [data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 2px rgba(139, 92, 246, .45); }

    /* Sidebar */
    [data-testid="stSidebar"] h2 { font-size: 1.1rem; }
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(139, 92, 246, .5); background: rgba(139, 92, 246, .05);
    }

    /* Primary-looking buttons get the portfolio gradient on hover */
    .stButton > button { border-radius: 999px; transition: all .2s ease; }
    .stButton > button:hover { border-color: var(--dm-violet); color: var(--dm-violet-light); }

    /* Sources */
    .source-box {
        background: rgba(139, 92, 246, .06); border-left: 3px solid var(--dm-violet);
        border-radius: .5rem; padding: .6rem .8rem; margin: .4rem 0; font-size: .85em;
    }
    .source-box strong { color: var(--dm-violet-light); }
    .source-box em { color: var(--dm-muted); }

    /* Scrollbar like the portfolio */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: var(--dm-violet); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# When the server has no OPENAI_API_KEY (e.g. a public demo), each visitor
# provides their own key. It only lives in this browser session's memory.
SERVER_API_KEY = os.getenv("OPENAI_API_KEY")


def get_api_key():
    """Return the OpenAI key to use, asking the visitor for one if needed."""
    if SERVER_API_KEY:
        return SERVER_API_KEY
    with st.sidebar:
        st.header("🔑 OpenAI API key")
        key = st.text_input(
            "Your OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Used only for this browser session, never stored. "
                 "Get one at https://platform.openai.com/api-keys",
        ).strip()
        st.caption("Your key stays in this session's memory and is sent only to OpenAI.")
    return key or None


def init_session(api_key):
    """Create (or recreate when the key changes) the per-session backend."""
    key_changed = st.session_state.get("api_key_in_use", api_key) != api_key
    if "documind" not in st.session_state or key_changed:
        st.session_state.documind = DocuMind(api_key=api_key)
        st.session_state.documents = st.session_state.documind.list_documents()
        st.session_state.messages = []
    st.session_state.api_key_in_use = api_key
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("documents", st.session_state.documind.list_documents())
    st.session_state.setdefault("uploader_key", 0)


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
    st.markdown("""
    <div class="dm-hero">
        <div class="dm-eyebrow">RAG System</div>
        <h1>🧠 Docu<span class="gradient-text">Mind</span></h1>
        <p>Chat with your documents. Every answer cites the file and page it comes from.</p>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_api_key()
    if not api_key:
        st.info("🔑 Enter your OpenAI API key in the sidebar to get started.")
        st.stop()
    init_session(api_key)

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
        st.markdown("""
        <div class="dm-cards">
            <div class="dm-card"><h4>📚 Research</h4><ul>
                <li>Analyze academic papers</li><li>Extract key findings</li><li>Compare methodologies</li>
            </ul></div>
            <div class="dm-card"><h4>💼 Business</h4><ul>
                <li>Query contracts</li><li>Summarize reports</li><li>Find specific clauses</li>
            </ul></div>
            <div class="dm-card"><h4>📖 Learning</h4><ul>
                <li>Study materials</li><li>Quick fact lookup</li><li>Concept explanations</li>
            </ul></div>
        </div>
        <div class="dm-tags">
            <span class="dm-tag">LangChain</span><span class="dm-tag">ChromaDB</span>
            <span class="dm-tag">GPT-4o-mini</span><span class="dm-tag">Streaming</span>
            <span class="dm-tag">Conversation memory</span>
        </div>
        """, unsafe_allow_html=True)
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
