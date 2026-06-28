from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import google.generativeai as genai
import math
import re
import json
import asyncio
from datetime import datetime, timedelta

app = FastAPI(title="News RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEWSAPI_BASE = "https://newsapi.org/v2/everything"


# ── Models ──────────────────────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    news_api_key: str
    topic: str
    language: str = "en"
    page_size: int = 20


class AskRequest(BaseModel):
    question: str
    articles: list[dict]
    gemini_api_key: Optional[str] = None


# ── TF-IDF helpers ───────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","as","that","this","it",
    "its","into","about","which","have","has","had","will","would","could",
    "should","not","also","more","their","they","he","she","we","you","i",
    "my","your","our","his","her","all","can","may","do","did","does","if",
    "so","up","out","no","than","then","these","those","when","where","how",
    "what","who","after","before","during","while","since","until","over",
    "just","even","new","one","two","three","said","says","say"
}


def tokenise(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
            if len(w) > 2 and w not in STOPWORDS]


def build_vocab(texts: list[str], max_terms: int = 400) -> list[str]:
    df: dict[str, int] = {}
    for t in texts:
        for w in set(tokenise(t)):
            df[w] = df.get(w, 0) + 1
    n = len(texts)
    return [w for w, c in sorted(df.items(), key=lambda x: -x[1])
            if 1 <= c < n * 0.85][:max_terms]


def tfidf_vector(text: str, vocab: list[str]) -> list[float]:
    words = tokenise(text)
    if not words:
        return [0.0] * len(vocab)
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [freq.get(term, 0) / len(words) for term in vocab]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, articles: list[dict], k: int = 6) -> list[dict]:
    texts = [a.get("_text", "") for a in articles]
    vocab = build_vocab(texts)
    q_vec = tfidf_vector(question, vocab)
    scored = []
    for a in articles:
        vec = tfidf_vector(a.get("_text", ""), vocab)
        scored.append({**a, "_score": cosine_sim(q_vec, vec)})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:k]


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fetch-news")
async def fetch_news(req: FetchRequest):
    today       = datetime.utcnow().date()
    recent_from = (datetime.utcnow() - timedelta(days=14)).date()
    old_from = (datetime.utcnow() - timedelta(days=28)).date()
    old_to   = (datetime.utcnow() - timedelta(days=15)).date()

    params_base = {
        "q": req.topic,
        "language": req.language,
        "pageSize": req.page_size,
        "sortBy": "relevancy",
        "apiKey": req.news_api_key,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        recent_resp, old_resp = await asyncio.gather(
            client.get(NEWSAPI_BASE, params={**params_base, "from": str(recent_from), "to": str(today)}),
            client.get(NEWSAPI_BASE, params={**params_base, "from": str(old_from),    "to": str(old_to)}),
        )

    recent_data = recent_resp.json()
    old_data    = old_resp.json()

    if recent_data.get("status") != "ok":
        raise HTTPException(400, detail=recent_data.get("message", "NewsAPI error (recent)"))
    if old_data.get("status") != "ok":
        raise HTTPException(400, detail=old_data.get("message", "NewsAPI error (older)"))

    def clean(raw_articles, era):
        out = []
        for a in raw_articles:
            if not a.get("title") or not a.get("description"):
                continue
            out.append({
                "title":       a["title"],
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
                "source":      a.get("source", {}).get("name", "Unknown"),
                "publishedAt": a.get("publishedAt", "")[:10],
                "era":         era,
                "_text":       a["title"] + " " + a.get("description", ""),
            })
        return out

    articles = (
        clean(recent_data.get("articles", []), "recent") +
        clean(old_data.get("articles", []),    "old")
    )

    if not articles:
        raise HTTPException(404, detail="No articles found. Try a different topic.")

    return {"articles": articles, "count": len(articles)}


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.articles:
        raise HTTPException(400, detail="No articles provided. Fetch news first.")
    if not req.gemini_api_key:
        raise HTTPException(400, detail="Gemini API key required.")

    top_docs = retrieve(req.question, req.articles, k=6)

    context_parts = []
    for i, a in enumerate(top_docs, 1):
        era_label = "Recent" if a.get("era") == "recent" else "Older"
        context_parts.append(
            f"[{i}] ({era_label} · {a.get('source','?')} · {a.get('publishedAt','')})\n"
            f"Title: {a['title']}\n"
            f"Summary: {a.get('description','')}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a precise news analyst. Answer the user's question using ONLY the provided articles. "
        "Cite articles with [1], [2], etc. Be concise and factual. "
        "Note temporal differences between recent and older articles where relevant.\n\n"
        f"Articles:\n{context}\n\n"
        f"Question: {req.question}\n\nAnswer:"
    )

    # Configure Gemini
    genai.configure(api_key=req.gemini_api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1000,
            temperature=0.3,
        ),
    )

    sources_payload = [
        {
            "index": i + 1,
            "title": a["title"],
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "publishedAt": a.get("publishedAt", ""),
            "era": a.get("era", ""),
        }
        for i, a in enumerate(top_docs)
    ]

    def stream_response():
        try:
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                text = chunk.text if hasattr(chunk, "text") else ""
                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True, 'sources': sources_payload})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")
