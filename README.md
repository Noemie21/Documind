---
title: DocuMind
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Chat with your PDF, Word and text documents (RAG)
---

# 🧠 DocuMind - AI Document Assistant

> **Chat with your documents using RAG (Retrieval-Augmented Generation)**

DocuMind is an intelligent document assistant that lets you ask questions about your PDF, Word, and text files. It uses semantic search to find relevant information and generates accurate answers with source citations.

[![CI](https://github.com/Noemie21/Documind/actions/workflows/ci.yml/badge.svg)](https://github.com/Noemie21/Documind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red.svg)

## ✨ Features

- **📄 Multi-format support** - PDF, DOCX, TXT files (any text encoding)
- **🔍 Semantic search** - Find relevant information even with different wording
- **📝 Source citations** - Every answer references the original file name and page
- **💬 Conversation memory** - Ask follow-ups like *"and what about the second point?"*
- **⚡ Streaming responses** - Answers appear token by token
- **📚 Multi-document** - Upload and query multiple documents at once
- **🔒 Session isolation** - Each browser session has its own private knowledge base

## 🔄 How a question is answered

1. **Condense** - If there is chat history, the follow-up question is rewritten into a standalone question
2. **Retrieve** - The 4 most similar chunks are fetched from this session's Chroma collection
3. **Generate** - GPT-4o-mini answers from those chunks (plus recent history) and streams the answer
4. **Cite** - The retrieved chunks are shown as sources under the answer

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│              Upload │ Chat │ Sources Display                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Document    │  │   Vector     │  │    LLM           │   │
│  │  Processing  │──│   Search     │──│    Generation    │   │
│  │  (Chunking)  │  │  (ChromaDB)  │  │   (GPT-4o-mini)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/Noemie21/Documind.git
cd Documind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run the app

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## 📖 Usage

1. **Upload documents** - Drag and drop PDF, DOCX, or TXT files in the sidebar
2. **Wait for processing** - Documents are chunked and embedded automatically
3. **Ask questions** - Type your question in the chat input
4. **Ask follow-ups** - The conversation is remembered, so you can refer to previous answers
5. **View sources** - Click on "Sources" to see where the information came from

> **Note:** documents are stored **in memory, per browser session**. Reloading the page or
> restarting the app starts from an empty knowledge base, and users never see each other's files.
> To keep a knowledge base on disk, create the backend with
> `DocuMind(persist_directory="./chroma_db", collection_name="my-kb")`.

### Example questions

- "What are the main findings of this research?"
- "Summarize the key points from chapter 3"
- "What does the contract say about termination?"
- "Compare the methodologies used in these papers"

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | LangChain | RAG pipeline orchestration |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic document representation |
| **Vector Store** | ChromaDB | Fast similarity search |
| **LLM** | GPT-4o-mini | Answer generation |
| **Frontend** | Streamlit | User interface |

## 📁 Project Structure

```
Documind/
├── app.py                  # RAG backend (DocumentProcessor, VectorStore, RAGChain, DocuMind)
├── streamlit_app.py        # Streamlit frontend
├── tests/                  # Pytest suite (offline: fake embeddings and LLM)
├── requirements.txt        # Pinned runtime dependencies
├── requirements-dev.txt    # + pytest, ruff
├── pyproject.toml          # pytest and ruff configuration
├── Dockerfile
├── .github/workflows/ci.yml
└── .env.example            # Environment variables template
```

## 🔧 Configuration

All settings are environment variables (put them in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `DOCUMIND_LLM_MODEL` | `gpt-4o-mini` | Chat model used for answers |
| `DOCUMIND_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `DOCUMIND_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DOCUMIND_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DOCUMIND_TOP_K` | `4` | Number of chunks retrieved per question |
| `DOCUMIND_MAX_HISTORY` | `6` | Chat messages kept as conversation memory |

## 🧪 Tests

The test suite runs fully offline (fake embeddings and a fake LLM, no API key needed):

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The same checks run on every push and pull request through GitHub Actions.

## 🚀 Deployment

### 🔑 API key: server key or visitor key

- If `OPENAI_API_KEY` is set on the server, everybody uses that key.
- If it is **not** set, each visitor is asked for **their own OpenAI key** in the sidebar.
  The key only lives in that browser session's memory. This is the recommended setup for a
  public demo, so strangers cannot spend your credits.

### Hugging Face Spaces (free, Docker)

The front matter at the top of this README is the Space configuration (`sdk: docker`, port 8501).

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space), SDK **Docker**, template **Blank**
2. Push this repository to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/documind
   git push space main
   ```
   (use a Hugging Face access token with *write* permission as the password)
3. Leave `OPENAI_API_KEY` unset for a public demo, or add it under *Settings → Variables and secrets*
4. If uploads fail with a 403 error inside the embedded page, add the variable
   `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false` in the Space settings

The app is then available at `https://<hf-username>-documind.hf.space`.

### Embed it on your own website

[`deploy/website/documind/index.html`](deploy/website/documind/index.html) is a standalone page that
presents the project and embeds the Space. Upload the `documind/` folder to the web root of any static
host (e.g. via FTP to `www/` on OVH shared hosting) to serve it at `https://your-domain/documind/`.

### Docker

```bash
docker build -t documind .
docker run -p 8501:8501 documind                              # visitors enter their own key
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... documind     # or use a server key
```

## 📈 Future Improvements

- [ ] Add support for more file formats (HTML, Markdown)
- [ ] Add hybrid search (semantic + keyword)
- [ ] Relevance threshold to avoid answering from unrelated chunks
- [ ] Answer-quality evaluation set (e.g. RAGAS)
- [ ] Remove a single document from the knowledge base
- [ ] Support for local LLMs (Ollama, vLLM)
- [ ] Add document comparison feature
- [ ] Implement reranking with Cohere

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this project for any purpose.

## 👤 Author

**Noémie K**
- GitHub: [@Noemie21](https://github.com/Noemie21)
- Website: [noemie.pro](http://noemie.pro)

---

⭐ If you found this project helpful, please give it a star!
