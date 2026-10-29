"""Где теряется нужный чанк: позиция на каждом этапе поиска графа.

Запуск: python tests/trace_query.py "как увеличить лимит задолженности" "нельзя"
Второй аргумент — кусок текста, который точно есть в нужном чанке (регистр не важен).
"""
import sys
from pathlib import Path

from qdrant_client import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.retrieval import (  # noqa: E402
    COLLECTION_NAME, RERANK_K, bm25, client, embed_query, find_candidates, rerank,
)

DEEP = 500  # насколько глубоко смотреть dense и sparse


def find_targets(needle):
    found, offset = [], None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=1000, offset=offset,
                                       with_payload=["chunk_id", "title", "content"])
        found += [p for p in points if needle in p.payload.get("content", "").lower()]
        if offset is None:
            return found


def rank(ids, target):
    return ids.index(target) + 1 if target in ids else None


def main():
    query, needle = sys.argv[1], sys.argv[2].lower()
    targets = find_targets(needle)
    print(f"Чанков с «{needle}»: {len(targets)}")
    if not targets:
        return

    dense = [p.id for p in client.query_points(COLLECTION_NAME, query=embed_query(query),
                                               using="dense", limit=DEEP).points]
    sp = next(iter(bm25.query_embed(query)))
    sparse = [p.id for p in client.query_points(
        COLLECTION_NAME, using="sparse", limit=DEEP,
        query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
    ).points] if len(sp.indices) else []
    cands = find_candidates(query)
    rrf = [c["chunk_id"] for c in cands]
    scored = rerank(query, cands)
    reranked = [p["chunk_id"] for _, p in scored]
    # скор реранкера для самих целевых чанков, даже если они не попали в кандидаты
    own = {p["chunk_id"]: s for s, p in rerank(query, [t.payload for t in targets])}
    top_score = scored[RERANK_K - 1][0] if len(scored) >= RERANK_K else None

    print(f"Порог входа в LLM (скор #{RERANK_K} после реранкера): {top_score}\n")
    print(f"{'chunk_id':<70}{'dense':>7}{'sparse':>8}{'RRF@50':>8}{'rerank':>8}{'скор':>8}")
    for t in targets:
        cid = t.payload["chunk_id"]
        cells = [rank(dense, t.id), rank(sparse, t.id), rank(rrf, cid), rank(reranked, cid)]
        print(f"{cid[:69]:<70}" + "".join(f"{c or '—':>{w}}" for c, w in zip(cells, (7, 8, 8, 8)))
              + f"{own[cid]:>8.2f}")

    print("\nЧто ушло в LLM вместо него:")
    for s, p in scored[:RERANK_K]:
        print(f"  {s:6.2f}  {p['chunk_id']}")


if __name__ == "__main__":
    main()
