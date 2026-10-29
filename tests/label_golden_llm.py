"""Первичная разметка relevant_ids в golden set с помощью GigaChat.

Запуск: python tests/label_golden_llm.py
Креды GigaChat берутся из .env в корне проекта (GIGACHAT_CREDENTIALS, GIGACHAT_SCOPE, ...).
Размечаются запросы с relevant_ids = null, прогресс сохраняется после каждого запроса.
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from gigachat import GigaChat
from qdrant_client import QdrantClient

from label_golden import (
    COLLECTION, GOLDEN_PATH, QDRANT_URL, TEXT_KEY, DOC_KEY, VECTOR_NAME, check_payload, embed,
)

ENV_PATH = Path(__file__).parent.parent / ".env"
MODEL = "GigaChat-3-Ultra"
CHUNK_LIMIT = 50       # чанков из dense и из BM25
DOC_LIMIT = 15         # документов-кандидатов на запрос
CHUNKS_PER_DOC = 3     # лучших чанков документа, которые видит судья
MAX_CHARS = 6000

PROMPT = """Ты оцениваешь, отвечает ли документ базы знаний МТС на вопрос клиента.

Вопрос клиента: {query}

Фрагменты документа:
---
{text}
---

Документ релевантен, только если во фрагментах есть информация, которая отвечает на вопрос полностью или на его существенную часть. Документ на ту же тему, но без ответа на этот вопрос — нерелевантен.

Ответь одним словом: ДА или НЕТ."""


def load_env():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / len(docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.k1, self.b = k1, b

    def top(self, query, k):
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        scores = []
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            scores.append(sum(self.idf[t] * d[t] * (self.k1 + 1) / (d[t] + norm) for t in terms if t in d))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


def load_bm25(client):
    payloads, offset = [], None
    while True:
        points, offset = client.scroll(COLLECTION, limit=1000, offset=offset, with_payload=True)
        payloads += [p.payload for p in points]
        if offset is None:
            break
    return BM25([tokenize(p[TEXT_KEY]) for p in payloads]), payloads


def doc_pool(client, bm25, payloads, query):
    dense = [p.payload for p in client.query_points(
        COLLECTION, query=embed(query), using=VECTOR_NAME, limit=CHUNK_LIMIT, with_payload=True
    ).points]
    sparse = [payloads[i] for i in bm25.top(query, CHUNK_LIMIT)]

    docs = {}
    for a, b in zip(dense, sparse):  # чередуем, чтобы оба источника попали в пул
        for payload in (a, b):
            for doc_id in payload.get("sources") or [payload[DOC_KEY]]:
                chunks = docs.setdefault(doc_id, [])
                if len(chunks) < CHUNKS_PER_DOC and payload[TEXT_KEY] not in chunks:
                    chunks.append(payload[TEXT_KEY])
    return list(docs.items())[:DOC_LIMIT]


def judge(giga, query, chunks):
    prompt = PROMPT.format(query=query, text="\n\n".join(chunks)[:MAX_CHARS])
    if hasattr(giga.chat, "create"):  # новый SDK
        text = giga.chat.create(prompt).messages[0].content[0].text
    else:                             # старый SDK
        text = giga.chat(prompt).choices[0].message.content
    return text.strip().upper().startswith("ДА")


def main():
    load_env()
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    client = QdrantClient(url=QDRANT_URL)
    check_payload(client)
    bm25, payloads = load_bm25(client)

    with GigaChat(model=MODEL, timeout=120, max_retries=5, retry_backoff_factor=2) as giga:
        for item in golden:
            if item["relevant_ids"] is not None:
                continue
            pool = doc_pool(client, bm25, payloads, item["query"])
            item["relevant_ids"] = [doc_id for doc_id, chunks in pool if judge(giga, item["query"], chunks)]
            print(f"[{item['id']}] {len(item['relevant_ids'])} док.: {item['relevant_ids']}")
            with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
                json.dump(golden, f, ensure_ascii=False, indent=2)

    done = sum(x["relevant_ids"] is not None for x in golden)
    print(f"\nРазмечено {done}/{len(golden)}")


if __name__ == "__main__":
    main()