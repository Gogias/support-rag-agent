"""chunk_first, этап 1: отбор абзацев, по которым потом пишутся вопросы.

Запуск: python tests/select_chunks.py
Берёт из коллекции B2C-чанки нормальной длины, не больше PER_FOLDER из одной папки
и не больше одного из одной статьи. Отдельно отбирает чанки с отрицательными
формулировками («нельзя», «не получится» и т.п.).
Результат: tests/golden_sets/chunk_first_candidates.json
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.retrieval import COLLECTION_NAME, client  # noqa: E402
from src.ingestion.segment import segment  # noqa: E402

OUT_PATH = Path(__file__).parent / "golden_sets" / "chunk_first_candidates.json"
N_NEGATIVE = 30
N_OTHER = 90
PER_FOLDER = 3       # из одной папки в каждую из двух подборок
MIN_CHARS, MAX_CHARS = 300, 2000   # длина текста без markdown-ссылок
SEED = 42

NEGATIVE = re.compile(r"\bнельзя\b|не получится|невозможно|недоступн|не предусмотрен|не может|не сможете"
                      r"|только в (салоне|офисе)")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def load_b2c():
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=1000, offset=offset, with_payload=[
            "chunk_id", "chunk_ids", "sources", "source_path", "title", "heading_path", "content"])
        for p in points:
            sources = p.payload.get("sources") or [p.payload["source_path"]]
            text = LINK.sub(r"\1", p.payload.get("content", ""))
            if any(segment(s) == "b2c" for s in sources) and MIN_CHARS <= len(text) <= MAX_CHARS:
                chunks.append({**p.payload, "sources": sources, "negative": bool(NEGATIVE.search(text.lower()))})
        if offset is None:
            return chunks


def sample(pool, n, used_docs):
    per_folder, picked = Counter(), []
    for c in pool:
        doc = c["sources"][0]
        folder = doc.split("/", 1)[0]
        if per_folder[folder] < PER_FOLDER and doc not in used_docs:
            picked.append(c)
            per_folder[folder] += 1
            used_docs.add(doc)
        if len(picked) == n:
            break
    return picked


def main():
    chunks = load_b2c()
    random.Random(SEED).shuffle(chunks)
    used_docs = set()
    negative = sample([c for c in chunks if c["negative"]], N_NEGATIVE, used_docs)
    other = sample([c for c in chunks if not c["negative"]], N_OTHER, used_docs)

    picked = negative + other
    for i, c in enumerate(picked, 1):
        c["cand_id"] = f"cand-{i:03d}"
    OUT_PATH.write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"B2C-чанков подходящей длины: {len(chunks)}, из них с отрицанием: {sum(c['negative'] for c in chunks)}")
    print(f"Отобрано: {len(negative)} с отрицанием + {len(other)} обычных = {len(picked)}")
    print(f"Папок: {len({c['sources'][0].split('/', 1)[0] for c in picked})}")
    print(f"Сохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
