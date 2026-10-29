"""Поиск по базе знаний: dense + sparse -> RRF -> реранкер -> статьи.

Ищем по абзацам (чанкам), а в LLM отдаём статьи: до MAX_ARTICLES лучших статей,
короткие целиком, длинные — разделами, где нашлись лучшие абзацы.
Ничего не знает про LangGraph: на входе строка запроса, на выходе список статей.
"""

import logging

import numpy as np
import requests
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

from src.config import settings
from src.ingestion.bm25 import make_bm25

log = logging.getLogger(__name__)

QDRANT_URL = settings.qdrant_url
EMBED_URL = settings.embed_url
COLLECTION_NAME = settings.qdrant_collection

RETRIEVE_K = 50
RERANK_URL = settings.rerank_url
RERANK_MAX_TOKENS = 1024   # <= -ub реранкера
RERANK_K = 5               # для eval-скриптов: сколько абзацев считать «топом» реранкера
RERANK_MIN_SCORE = float("-inf")  # статьи с лучшим абзацем ниже порога не уходят в LLM; откалибровать позже, если надо
MAX_ARTICLES = 5           # сколько статей уходит в LLM
ARTICLE_MAX_CHARS = 5000   # статья короче — целиком (97% B2C-статей), длиннее — нужные разделы
CONTEXT_CHARS = 12000      # общий бюджет контекста, ~4000 токенов
B2C = models.Filter(must=[models.FieldCondition(key="segments", match=models.MatchValue(value="b2c"))])

client = QdrantClient(url=QDRANT_URL)
http = requests.Session()
bm25 = make_bm25()
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)


def embed_query(text: str) -> list[float]:
    # Та же нормализация, что и при индексации
    resp = http.post(EMBED_URL, json={"content": text}, timeout=60)
    resp.raise_for_status()
    emb = np.asarray(resp.json()[0]["embedding"][0], dtype=np.float32)
    return (emb / np.linalg.norm(emb)).tolist()


def find_candidates(query: str) -> list[dict]:
    dense = embed_query(query)
    sp = next(iter(bm25.query_embed(query)))
    prefetch = [models.Prefetch(query=dense, using="dense", limit=RETRIEVE_K, filter=B2C)]
    if len(sp.indices):
        sparse = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
        prefetch.append(models.Prefetch(query=sparse, using="sparse", limit=RETRIEVE_K, filter=B2C))
    points = client.query_points(COLLECTION_NAME, prefetch=prefetch,
                                 query=models.FusionQuery(fusion=models.Fusion.RRF),
                                 limit=RETRIEVE_K, with_payload=True).points
    return [p.payload for p in points]


def rerank(query: str, payloads: list[dict]) -> list[tuple[float, dict]]:
    budget = RERANK_MAX_TOKENS - len(tokenizer.encode(query, add_special_tokens=False)) - 16
    docs = []
    for p in payloads:
        text = f"{p.get('title', '')}\n{' > '.join(p.get('heading_path') or [])}\n{p.get('content', '')}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        docs.append(text if len(ids) <= budget else tokenizer.decode(ids[:budget]))
    resp = http.post(RERANK_URL, json={"query": query, "documents": docs}, timeout=120)
    resp.raise_for_status()
    scores = {r["index"]: r["relevance_score"] for r in resp.json()["results"]}
    return sorted(((scores[i], p) for i, p in enumerate(payloads)), key=lambda x: x[0], reverse=True)


def own_chunk_id(payload: dict, source: str) -> str:
    """chunk_id абзаца именно в этой статье: у дедуплицированной точки их несколько."""
    return next((c for c in payload.get("chunk_ids") or [] if c.startswith(source + "::")), payload["chunk_id"])


def article_chunks(source: str) -> list[dict]:
    """Все абзацы статьи по порядку. Абзац-дубль может лежать под путём другой статьи, поэтому ищем и по sources."""
    points, _ = client.scroll(COLLECTION_NAME, limit=500, with_payload=True, scroll_filter=models.Filter(should=[
        models.FieldCondition(key="source_path", match=models.MatchValue(value=source)),
        models.FieldCondition(key="sources", match=models.MatchValue(value=source)),
    ]))
    return sorted((p.payload for p in points), key=lambda p: int(own_chunk_id(p, source).rsplit("::", 1)[1]))


def fit(chunks: list[dict], hits: list, limit: int) -> list[dict]:
    """Что отдать от статьи в пределах limit символов: целиком → нужные разделы → лучшие абзацы.
    Лучший абзац статьи не теряется никогда: если он один не влезает, отдаётся обрезанным."""
    limit = min(limit, ARTICLE_MAX_CHARS)
    size = lambda cs: sum(len(c["content"]) for c in cs)  # noqa: E731
    if size(chunks) <= limit:
        return chunks

    best = [p["chunk_id"] for _, p in hits[:3]]  # лучшие абзацы статьи в порядке скора
    sections = {tuple(c.get("heading_path") or [])[:1] for c in chunks if c["chunk_id"] in best}
    part = [c for c in chunks if tuple(c.get("heading_path") or [])[:1] in sections]
    if size(part) <= limit:
        return part

    by_id = {c["chunk_id"]: c for c in chunks}
    picked, total = set(), 0
    for cid in best:  # по одному в порядке скора, пока влезают
        c = by_id.get(cid)
        if c and total + len(c["content"]) <= limit:
            picked.add(cid)
            total += len(c["content"])
    if picked:
        return [c for c in chunks if c["chunk_id"] in picked]  # в порядке статьи

    top = by_id.get(best[0]) if best else None
    return [{**top, "content": top["content"][:limit]}] if top and limit >= 500 else []


def search(query: str) -> list[dict]:
    """Ищем по абзацам, отдаём статьи: до MAX_ARTICLES статей с лучшими абзацами."""
    payloads = find_candidates(query)
    log.info("кандидатов из Qdrant: %d", len(payloads))
    if not payloads:
        return []
    try:
        ranked = rerank(query, payloads)
        log.info("топ скоры реранкера: %s", [round(s, 2) for s, _ in ranked[:RERANK_K]])
    except requests.RequestException as e:
        log.warning("реранкер недоступен (%s), порядок по RRF", e)
        ranked = [(None, p) for p in payloads]

    # статьи в порядке их лучшего абзаца (ranked уже отсортирован по скору)
    articles = {}
    for s, p in ranked:
        articles.setdefault((p.get("sources") or [p["source_path"]])[0], []).append((s, p))

    results, used = [], 0
    for source, hits in articles.items():
        score = hits[0][0]
        if score is not None and score < RERANK_MIN_SCORE:
            break
        chunks = article_chunks(source)
        # первая статья — с полным бюджетом, следующие — с тем, что осталось
        chunks = fit(chunks, hits, CONTEXT_CHARS - used) if chunks else []
        if not chunks:
            continue
        text = "\n\n".join(c["content"] for c in chunks)
        results.append({
            "title": chunks[0].get("title") or "",
            "heading_path": [],
            "content": text,
            "before": "",
            "after": "",
            "sources": [source],
            "score": score,
            "chunk_ids": [cid for c in chunks for cid in (c.get("chunk_ids") or [c["chunk_id"]])],
        })
        used += len(text)
        if len(results) == MAX_ARTICLES:
            break

    log.info("в LLM: %s", [(r["sources"][0], None if r["score"] is None else round(r["score"], 2), len(r["content"]))
                           for r in results])
    return results