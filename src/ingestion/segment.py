"""Сегмент статьи по пути источника: b2c, b2b или bank.

Нужен метрике OffSegment (tests/eval_retrieval.py), разметке (tests/relabel_golden.py)
и фильтру поиска (поле segment в payload Qdrant).

Запуск: python -m src.ingestion.segment — список папок коллекции с сегментом и числом чанков,
чтобы один раз глазами проверить префиксы ниже.
"""
B2B = ("lk_mts_business", "mobilnaya-svyaz-b2b", "mts-profil-dlya-biznesa", "mts_marketolog", "mts_dsp",
       "segmento", "omni_retail_product", "mts_omnichannel", "mts_iot_hub",
       "mts_domofon_dlya_yuridicheskih_lits", "mws_", "msphere", "mts-tenders",
       "mobilnie_sotrudniki", "mts_koordinator", "mts_dealer_life", "mts_partner", "partnerskaya-set", "mts_agents", "avtosekretar", "mts_ekologiya")
BANK = ("karti", "vkladi_i_scheta", "krediti_i", "invetitsii", "platezhi_i_perevodi",
        "strahovanie", "mts_cashback", "mts_nakopleniya")
BANK_IN = ("mts_dengi", "mts_bank")  # банковский маркер в любом месте имени папки


def segment(source_path: str) -> str:
    folder = source_path.split("/", 1)[0]
    if folder.startswith(BANK) or any(m in folder for m in BANK_IN):
        return "bank"
    if folder.startswith(B2B):
        return "b2b"
    return "b2c"


if __name__ == "__main__":
    from collections import Counter

    from qdrant_client import QdrantClient

    from src.config import settings

    client = QdrantClient(url=settings.qdrant_url)
    folders, mixed, offset = Counter(), 0, None
    while True:
        points, offset = client.scroll(settings.qdrant_collection, limit=1000, offset=offset,
                                       with_payload=["sources", "source_path"])
        for p in points:
            sources = p.payload.get("sources") or [p.payload["source_path"]]
            folders.update(s.split("/", 1)[0] for s in sources)
            mixed += len({segment(s) for s in sources}) > 1
        if offset is None:
            break

    for folder, n in sorted(folders.items(), key=lambda x: (segment(x[0]), x[0])):
        print(f"{segment(folder):5} {n:>6}  {folder}")
    print(f"\nЧанков-дублей с источниками из разных сегментов: {mixed}")