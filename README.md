# 🧠 DocuMind - AI Document Assistant

> **Chat with your documents using RAG (Retrieval-Augmented Generation)**

DocuMind is an intelligent document assistant that lets you ask questions about your PDF, Word, and text files. It uses semantic search to find relevant information and generates accurate answers with source citations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)

## ✨ Features

- **📄 Multi-format support** - PDF, DOCX, TXT files
- **🔍 Semantic search** - Find relevant information even with different wording
- **📝 Source citations** - Every answer includes references to the original document
- **💬 Chat interface** - Natural conversation with your documents
- **📚 Multi-document** - Upload and query multiple documents at once
- **⚡ Fast responses** - Optimized chunking and retrieval

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

- Python 3.9+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/documind.git
cd documind

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
4. **View sources** - Click on "Sources" to see where the information came from

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
documind/
├── app.py              # RAG backend (DocumentProcessor, VectorStore, RAGChain)
├── streamlit_app.py    # Streamlit frontend
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## 🔧 Configuration

You can customize the RAG pipeline by modifying these parameters in `app.py`:

```python
# Chunking parameters
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks

# Retrieval parameters
k = 4                  # Number of chunks to retrieve

# LLM parameters
model = "gpt-4o-mini"  # OpenAI model to use
temperature = 0        # Response randomness (0 = deterministic)
```

## 🚀 Deployment

### Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload all files
4. Add your `OPENAI_API_KEY` in Settings > Secrets

### Docker

```bash
docker build -t documind .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key documind
```

## 📈 Future Improvements

- [ ] Add support for more file formats (HTML, Markdown)
- [ ] Implement conversation memory
- [ ] Add hybrid search (semantic + keyword)
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
