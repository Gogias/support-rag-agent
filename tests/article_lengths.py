"""Длины B2C-статей в коллекции: сколько символов и чанков, хватит ли контекста отдавать статью целиком."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.retrieval import COLLECTION_NAME, client  # noqa: E402
from src.ingestion.segment import segment  # noqa: E402

chars, chunks, offset = defaultdict(int), defaultdict(int), None
while True:
    points, offset = client.scroll(COLLECTION_NAME, limit=1000, offset=offset,
                                   with_payload=["sources", "source_path", "content"])
    for p in points:
        src = (p.payload.get("sources") or [p.payload["source_path"]])[0]
        if segment(src) == "b2c":
            chars[src] += len(p.payload.get("content", ""))
            chunks[src] += 1
    if offset is None:
        break

lengths = sorted(chars.values())
n = len(lengths)
print(f"B2C-статей: {n}, чанков: {sum(chunks.values())}")
for q in (50, 75, 90, 95, 99):
    print(f"  {q}-й перцентиль: {lengths[int(n * q / 100) - 1]:>7} символов")
for limit in (3000, 5000, 8000, 12000):
    print(f"  статей не длиннее {limit:>5} символов: {sum(x <= limit for x in lengths) / n:.0%}")
print("\nСамые длинные:")
for src in sorted(chars, key=chars.get, reverse=True)[:10]:
    print(f"  {chars[src]:>7}  {chunks[src]:>3} чанков  {src}")