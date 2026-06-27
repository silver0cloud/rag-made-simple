# News RAG Pipeline

Retrieve-Augment-Generate pipeline over live news. Fetches recent (last 30 days) and older (60–90 days) articles from NewsAPI, indexes them with TF-IDF, retrieves the most relevant context for each question, and streams answers via Claude.

```
news-rag/
├── backend/
│   ├── main.py              # FastAPI app — fetch, index, retrieve, stream
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── ConfigPanel.jsx   # API keys + topic config
    │   │   ├── FetchPanel.jsx    # Fetch & display article chips
    │   │   ├── QuestionPanel.jsx # Question input + suggestions
    │   │   └── AnswerPanel.jsx   # Streaming answer + sources
    │   └── index.css
    ├── index.html
    ├── vite.config.js
    └── package.json
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- [NewsAPI key](https://newsapi.org/register) (free tier: 100 req/day)
- [Gemini API key (https://aistudio.google.com/app/apikey)

---

## Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) copy env file if you want server-side keys
cp .env.example .env

# Start the server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Frontend setup

```bash
cd frontend

npm install
npm run dev
```

App runs at `http://localhost:5173`. The Vite dev server proxies `/api/*` → `http://localhost:8000`.

---

## How it works

### 1. Fetch & index (`POST /fetch-news`)
Two parallel NewsAPI requests:
- **Recent** — last 30 days, sorted by relevancy
- **Older** — 60 to 90 days ago, sorted by relevancy

Articles are returned as JSON with a `_text` field (title + description concatenated) ready for vectorisation.

### 2. Retrieval (client-side state, server-side on `/ask`)
The backend receives the full article list with each `/ask` request and:
1. Builds a TF-IDF vocabulary from all article texts
2. Vectorises the user's question and every article
3. Ranks by cosine similarity, picks the top 6

> **Scaling tip:** For production, replace in-memory TF-IDF with a vector database (Chroma, Qdrant, Pinecone). Add sentence-transformer embeddings for much better retrieval quality.

### 3. Generate (`POST /ask` → SSE stream)
The top-6 articles are injected as numbered context into a Claude prompt. The response streams back as Server-Sent Events (`text/event-stream`) and is displayed token-by-token in the UI. The final SSE event includes source metadata.

---

## API reference

### `POST /fetch-news`
```json
{
  "news_api_key": "...",
  "topic": "artificial intelligence",
  "language": "en",
  "page_size": 20
}
```
Returns `{ "articles": [...], "count": N }`.

### `POST /ask`
```json
{
  "question": "What are the latest breakthroughs?",
  "articles": [...],
  "gemini_api_key": "AIza..."
}
```
Returns an SSE stream. Each event is `data: {"text": "..."}`. The final event is `data: {"done": true, "sources": [...]}`.

---

## Extending the pipeline

| Goal | Change |
|------|--------|
| Better retrieval | Swap TF-IDF for sentence-transformers + Chroma |
| Persist articles | Store fetched articles in SQLite / Postgres |
| Multi-topic | Add a topic selector and per-topic article stores |
| Authentication | Add FastAPI `Depends` with JWT or API key header |
| Deploy | Dockerfile included below |

### Minimal Dockerfile (backend)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
