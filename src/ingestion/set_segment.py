"""Записывает в payload каждой точки поле segments — сегменты всех её источников — и создаёт индекс.

Запуск из корня проекта: python -m src.ingestion.set_segment
Один раз; после переиндексации (chunk_and_embed.py) запустить снова.
segments — список: у склеенного дубля могут быть копии из разных сегментов,
фильтр по массиву пропускает точку, если в нём есть нужный сегмент.
"""
from collections import Counter

from qdrant_client import QdrantClient, models

from src.config import settings
from src.ingestion.segment import segment

COLLECTION_NAME = settings.qdrant_collection

client = QdrantClient(url=settings.qdrant_url)
by_segments, offset = {}, None
while True:
    points, offset = client.scroll(COLLECTION_NAME, limit=1000, offset=offset, with_payload=["sources", "source_path"])
    for p in points:
        sources = p.payload.get("sources") or [p.payload["source_path"]]
        by_segments.setdefault(tuple(sorted({segment(s) for s in sources})), []).append(p.id)
    if offset is None:
        break

for segs, ids in by_segments.items():
    for i in range(0, len(ids), 1000):
        client.set_payload(COLLECTION_NAME, payload={"segments": list(segs)}, points=ids[i:i + 1000])

client.create_payload_index(COLLECTION_NAME, field_name="segments", field_schema=models.PayloadSchemaType.KEYWORD)

counts = Counter()
for segs, ids in by_segments.items():
    counts[" + ".join(segs)] += len(ids)
for segs, n in counts.most_common():
    print(f"{n:>6}  {segs}")