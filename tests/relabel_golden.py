"""Переразметка golden set: градуированная релевантность 0–3 от GigaChat (по схеме UMBRELA).

Запуск: python tests/relabel_golden.py
Читает  tests/golden_sets/golden_set_MTS_v3.json (запросы),
пишет   tests/golden_sets/golden_set_MTS_v4.json (новая разметка), прогресс сохраняется после каждого запроса.

Для каждого запроса:
  1. Пул кандидатов из поиска графа (src/agent/retrieval.py): top-10 после реранкера + top-5 dense + top-5 sparse.
  2. GigaChat оценивает каждый чанк: 3 — полный ответ, 2 — частичный, 1 — по теме без ответа, 0 — не по теме.
  3. Оценка документа = лучшая оценка его чанков; relevant_ids = документы с оценкой >= REL_MIN.
Обоснования судьи сохраняются в judgments — по ним удобно проверять разметку руками.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from gigachat import GigaChat
from qdrant_client import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.retrieval import (  # noqa: E402
    COLLECTION_NAME, bm25, client, embed_query, find_candidates, rerank,
)

SRC_PATH = Path(__file__).parent / "golden_sets" / "golden_set_MTS_v3.json"
OUT_PATH = Path(__file__).parent / "golden_sets" / "golden_set_MTS_v4.json"
ENV_PATH = ROOT / ".env"
MODEL = "GigaChat-3-Ultra"
POOL = {"rerank": 10, "dense": 5, "sparse": 5}  # сколько чанков берём из каждого источника
REL_MIN = 2           # с какой оценки документ считается релевантным
MAX_CHARS = 3000      # обрезка текста чанка для судьи

PROMPT = """Ты — асессор качества поиска службы поддержки МТС.
Клиент — частное лицо, абонент МТС (мобильная связь, домашний интернет и ТВ, подписки и сервисы МТС).
Документы для корпоративных клиентов (B2B), МТС Банка и МТС Деньги ему не подходят, если вопрос не о них.

Вопрос клиента: {query}

Фрагмент базы знаний:
Документ: {source}
Раздел: {path}
---
{content}
---

Действуй по шагам:
1. Сформулируй, что на самом деле хочет узнать или сделать клиент.
2. Проверь, есть ли во фрагменте информация, которая отвечает именно на это, а не просто относится к той же теме.
3. Поставь оценку:
3 — фрагмент прямо и полностью отвечает на вопрос;
2 — фрагмент отвечает на вопрос частично или на его главную часть;
1 — фрагмент на ту же тему, но ответа на вопрос в нём нет;
0 — фрагмент не относится к вопросу.

Ответь строго в формате:
Намерение: <одна фраза>
Обоснование: <одна-две фразы>
Оценка: <0, 1, 2 или 3>"""


def load_env():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def dense(query, k):
    return [p.payload for p in client.query_points(
        COLLECTION_NAME, query=embed_query(query), using="dense", limit=k, with_payload=True
    ).points]


def sparse(query, k):
    sp = next(iter(bm25.query_embed(query)))
    if not len(sp.indices):
        return []
    return [p.payload for p in client.query_points(
        COLLECTION_NAME, query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
        using="sparse", limit=k, with_payload=True,
    ).points]


def pool(query):
    reranked = [p for _, p in rerank(query, find_candidates(query))][:POOL["rerank"]]
    chunks = {}
    for p in reranked + dense(query, POOL["dense"]) + sparse(query, POOL["sparse"]):
        chunks.setdefault(p.get("chunk_id") or p["content_hash"], p)
    return list(chunks.values())


def judge(giga, query, p):
    prompt = PROMPT.format(
        query=query,
        source=", ".join(p.get("sources") or [p["source_path"]]),
        path=" > ".join([p.get("title") or "", *(p.get("heading_path") or [])]),
        content=p.get("content", "")[:MAX_CHARS],
    )
    if hasattr(giga.chat, "create"):  # новый SDK
        text = giga.chat.create(prompt).messages[0].content[0].text
    else:                             # старый SDK
        text = giga.chat(prompt).choices[0].message.content
    grades = re.findall(r"Оценка:\s*([0-3])", text)
    reason = re.search(r"Обоснование:\s*(.+)", text)
    return (int(grades[-1]) if grades else 0), (reason.group(1).strip() if reason else text.strip()[:300])


def main():
    load_env()
    with open(SRC_PATH, encoding="utf-8") as f:
        items = json.load(f)
    done = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            done = {x["id"]: x for x in json.load(f) if "relevance" in x}

    result = []
    with GigaChat(model=MODEL, timeout=120, max_retries=5, retry_backoff_factor=2) as giga:
        for i, item in enumerate(items, 1):
            if item["id"] in done:
                result.append(done[item["id"]])
                continue
            t = time.perf_counter()
            candidates = pool(item["query"])
            print(f"[{i}/{len(items)}] {item['id']}: {len(candidates)} чанков ", end="", flush=True)

            relevance, judgments = {}, []
            for p in candidates:
                grade, reason = judge(giga, item["query"], p)
                print(grade, end="", flush=True)
                sources = p.get("sources") or [p["source_path"]]
                for doc in sources:
                    relevance[doc] = max(grade, relevance.get(doc, 0))
                if grade > 0:
                    judgments.append({"chunk_id": p.get("chunk_id"), "sources": sources,
                                      "grade": grade, "reason": reason})

            relevance = dict(sorted(((d, g) for d, g in relevance.items() if g > 0), key=lambda x: -x[1]))
            result.append({
                **{k: item[k] for k in ("id", "query", "product", "section")},
                "relevant_ids": [d for d, g in relevance.items() if g >= REL_MIN],
                "relevance": relevance,
                "judgments": sorted(judgments, key=lambda x: -x["grade"]),
                "source": item.get("source", "synthetic"),
                "verified": False,
            })
            print(f"  → {len(result[-1]['relevant_ids'])} релевантных, {time.perf_counter() - t:.0f} с")
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(result + [x for x in items[i:] if x["id"] in done], f, ensure_ascii=False, indent=2)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    empty = [x["id"] for x in result if not x["relevant_ids"]]
    print(f"\nГотово: {OUT_PATH}")
    print(f"Без релевантных документов (проверить руками): {', '.join(empty) or 'нет'}")


if __name__ == "__main__":
    main()
