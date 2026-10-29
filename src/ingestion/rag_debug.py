"""rag_debug.py — интерактивная отладка RAG.

Вопрос -> dense + sparse -> RRF -> реранкер -> соседние чанки -> Qwen (KoboldCpp).
Каждый этап печатается, чтобы было видно, откуда взялся ответ.

python rag_debug.py                        # интерактивный режим
python rag_debug.py "как отключить подписку"   # один вопрос
python rag_debug.py --context              # + печать полного контекста, ушедшего в LLM
"""

import argparse
import time

import requests
from openai import OpenAI
from qdrant_client import models
from transformers import AutoTokenizer

from hybrid_search import COLLECTION_NAME, bm25, client, embed_query

# ---- Конфиг ------------------------------------------------------------------
RETRIEVE_K = 50            # сколько берём из dense, sparse и после RRF
SHOW_RETRIEVED = 15        # сколько строк поиска печатать
RERANK_URL = "http://127.0.0.1:5003/v1/rerank"
RERANK_MAX_TOKENS = 1024   # <= -ub реранкера
RERANK_K = 5               # сколько чанков идёт в LLM
RERANK_MIN_SCORE = -2.0    # ниже — отсекаем как нерелевантное
NEIGHBOR_MAX_CHARS = 800   # обрезка соседних чанков в контексте

KOBOLD_URL = "http://localhost:5001/v1"
LLM_MODEL = "qwen3.5"
LLM_MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "Ты — ассистент техподдержки МТС. Отвечай на вопрос пользователя ТОЛЬКО по "
    "фрагментам документации из контекста. Каждое утверждение помечай номером "
    "фрагмента в квадратных скобках, например [2]. Если в контексте нет ответа, "
    "прямо скажи, что в документации этого нет, и ничего не придумывай. "
    "Отвечай на русском, кратко и по шагам, если это инструкция."
)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
llm = OpenAI(base_url=KOBOLD_URL, api_key="not-needed", timeout=300)


def n_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def label(payload: dict, width: int = 75) -> str:
    path = " > ".join(payload.get("heading_path") or [])
    text = payload.get("title") or "?"
    if path:
        text += f" > {path}"
    return text if len(text) <= width else text[:width - 1] + "…"


def header(title: str, seconds: float | None = None) -> None:
    suffix = f"  ({seconds:.2f} с)" if seconds is not None else ""
    print(f"\n{'═' * 4} {title}{suffix} {'═' * max(0, 70 - len(title) - len(suffix))}")


# ---- 1. Поиск ------------------------------------------------------------------
def retrieve(query: str):
    dense = embed_query(query)
    sp = next(iter(bm25.query_embed(query)))
    sparse = (models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
              if len(sp.indices) else None)

    dense_pts = client.query_points(COLLECTION_NAME, query=dense, using="dense",
                                    limit=RETRIEVE_K).points
    sparse_pts = (client.query_points(COLLECTION_NAME, query=sparse, using="sparse",
                                      limit=RETRIEVE_K).points if sparse else [])

    prefetch = [models.Prefetch(query=dense, using="dense", limit=RETRIEVE_K)]
    if sparse:
        prefetch.append(models.Prefetch(query=sparse, using="sparse", limit=RETRIEVE_K))
    fused = client.query_points(COLLECTION_NAME, prefetch=prefetch,
                                query=models.FusionQuery(fusion=models.Fusion.RRF),
                                limit=RETRIEVE_K, with_payload=True).points

    d_rank = {p.id: i + 1 for i, p in enumerate(dense_pts)}
    s_rank = {p.id: i + 1 for i, p in enumerate(sparse_pts)}
    return fused, d_rank, s_rank, len(sp.indices)


# ---- 2. Реранкер ---------------------------------------------------------------
def fit_to_reranker(query: str, passage: str) -> str:
    budget = RERANK_MAX_TOKENS - n_tokens(query) - 16
    ids = tokenizer.encode(passage, add_special_tokens=False)
    return passage if len(ids) <= budget else tokenizer.decode(ids[:budget])


def rerank(query: str, points) -> list[tuple[float, object]]:
    passages = [
        fit_to_reranker(query, f"{p.payload.get('title', '')}\n"
                               f"{' > '.join(p.payload.get('heading_path') or [])}\n"
                               f"{p.payload.get('content', '')}")
        for p in points
    ]
    resp = requests.post(RERANK_URL, json={"query": query, "documents": passages}, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"реранкер {resp.status_code}: {resp.text[:300]}")
    scores = {r["index"]: r["relevance_score"] for r in resp.json()["results"]}
    return sorted(((scores[i], p) for i, p in enumerate(points)), key=lambda x: x[0], reverse=True)


# ---- 3. Контекст с соседями ---------------------------------------------------
def neighbors(payload: dict) -> tuple[str, str]:
    prev_id, next_id = payload.get("prev_chunk_id"), payload.get("next_chunk_id")
    wanted = [x for x in (prev_id, next_id) if x]
    found = {}
    if wanted:
        pts, _ = client.scroll(
            COLLECTION_NAME,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="chunk_ids", match=models.MatchAny(any=wanted)),
            ]),
            limit=len(wanted), with_payload=["chunk_ids", "content"],
        )
        for p in pts:
            for cid in p.payload["chunk_ids"]:
                if cid in wanted:
                    found[cid] = p.payload["content"][:NEIGHBOR_MAX_CHARS]
    return found.get(prev_id, ""), found.get(next_id, "")


def build_context(selected) -> str:
    blocks = []
    for n, (_, p) in enumerate(selected, 1):
        before, after = neighbors(p.payload)
        parts = [f"[{n}] {label(p.payload, width=200)}"]
        if before:
            parts.append(f"(предыдущий фрагмент) {before}")
        parts.append(p.payload.get("content", ""))
        if after:
            parts.append(f"(следующий фрагмент) {after}")
        blocks.append("\n".join(parts))
    return "\n\n---\n\n".join(blocks)


# ---- 4. LLM ------------------------------------------------------------------------
def answer(query: str, context: str) -> None:
    stream = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=LLM_MAX_TOKENS,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ---- Один вопрос -----------------------------------------------------------------
def run(query: str, show_context: bool) -> None:
    t = time.perf_counter()
    fused, d_rank, s_rank, n_terms = retrieve(query)
    header("1. ПОИСК", time.perf_counter() - t)
    print(f"dense: {len(d_rank)} | sparse: {len(s_rank)} (термов в запросе после стемминга: {n_terms}) "
          f"| после RRF: {len(fused)}")
    print(f"{'RRF':>4} {'D#':>4} {'S#':>4}  статья > раздел")
    for i, p in enumerate(fused[:SHOW_RETRIEVED], 1):
        print(f"{i:>4} {d_rank.get(p.id, '—'):>4} {s_rank.get(p.id, '—'):>4}  {label(p.payload)}")

    t = time.perf_counter()
    ranked = rerank(query, fused)
    header("2. РЕРАНКЕР", time.perf_counter() - t)
    rrf_pos = {p.id: i + 1 for i, p in enumerate(fused)}
    selected = [(s, p) for s, p in ranked[:RERANK_K] if s >= RERANK_MIN_SCORE]
    selected_ids = {p.id for _, p in selected}
    print(f"{'#':>3} {'score':>7} {'RRF#':>5}  статья > раздел")
    for i, (score, p) in enumerate(ranked[:RERANK_K + 5], 1):
        mark = "→" if p.id in selected_ids else " "
        print(f"{mark}{i:>2} {score:>7.3f} {rrf_pos[p.id]:>5}  {label(p.payload)}")
    print(f"→ в LLM: {len(selected)} (top-{RERANK_K}, порог score ≥ {RERANK_MIN_SCORE})")

    if not selected:
        header("3. ОТВЕТ")
        print("Нет ни одного чанка выше порога — в документации, вероятно, нет ответа.")
        return

    t = time.perf_counter()
    context = build_context(selected)
    header("3. КОНТЕКСТ", time.perf_counter() - t)
    print(f"{n_tokens(context)} токенов в {len(selected)} фрагментах")
    if show_context:
        print(context)

    t = time.perf_counter()
    header(f"4. ОТВЕТ ({LLM_MODEL})")
    answer(query, context)
    print(f"({time.perf_counter() - t:.1f} с)")

    header("ИСТОЧНИКИ")
    for n, (_, p) in enumerate(selected, 1):
        print(f"[{n}] {', '.join(p.payload.get('sources') or [p.payload.get('source_path')])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--context", action="store_true", help="печатать полный контекст для LLM")
    args = parser.parse_args()

    if args.question:
        run(" ".join(args.question), args.context)
        return
    while True:
        query = input("\nВопрос (пусто — выход)> ").strip()
        if not query:
            break
        try:
            run(query, args.context)
        except Exception as exc:  # noqa: BLE001 — сессия отладки не должна падать
            print(f"\n✗ {exc}")


if __name__ == "__main__":
    main()
