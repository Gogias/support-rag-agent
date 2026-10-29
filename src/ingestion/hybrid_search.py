"""Гибридный поиск: dense (BGE-M3 через llama-server) + sparse (BM25) -> RRF."""

import numpy as np
import requests
from qdrant_client import QdrantClient, models

from bm25 import make_bm25

QDRANT_URL = "http://localhost:6333"
LLAMA_SERVER_URL = "http://localhost:5002/embedding"
COLLECTION_NAME = "MTS_support_v2"

client = QdrantClient(url=QDRANT_URL)
bm25 = make_bm25()  # avg_len для запросов не нужен


def embed_query(text: str) -> list[float]:
    # Та же нормализация, что и при индексации
    resp = requests.post(LLAMA_SERVER_URL, json={"content": text}, timeout=60)
    resp.raise_for_status()
    emb = np.asarray(resp.json()[0]["embedding"][0], dtype=np.float32)
    return (emb / np.linalg.norm(emb)).tolist()


def hybrid_search(query: str, limit: int = 100, query_filter: models.Filter | None = None):
    prefetch = [
        models.Prefetch(query=embed_query(query), using="dense", limit=limit, filter=query_filter),
    ]

    sp = next(iter(bm25.query_embed(query)))  # query_embed для запроса, не embed
    if len(sp.indices) > 0:  # запрос из одних стоп-слов -> только dense
        prefetch.append(models.Prefetch(
            query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
            using="sparse",
            limit=limit,
            filter=query_filter,  # фильтр — внутрь КАЖДОГО prefetch
        ))

    return client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


def sparse_only(query: str, limit: int = 10):
    sp = next(iter(bm25.query_embed(query)))
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
        using="sparse",
        limit=limit,
        with_payload=True,
    ).points


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "как отключить платные подписки"
    print("== sparse only ==")
    for p in sparse_only(q, 5):
        print(f"{p.score:.3f}  {p.payload['title']}")
    print("== hybrid ==")
    for p in hybrid_search(q, 5):
        print(f"{p.score:.4f}  {p.payload['title']}")
