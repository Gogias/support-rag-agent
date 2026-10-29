"""Единая точка создания BM25-энкодера для индексации и поиска.

avg_len влияет только на веса ДОКУМЕНТОВ (нормализация по длине).
query_embed() его не использует — у терма запроса вес всегда 1.0,
а IDF добавляет сам Qdrant (Modifier.IDF). Поэтому при поиске важно
совпадение только модели и языка (стеммер + стоп-слова).
"""

from pathlib import Path

from fastembed import SparseTextEmbedding

BM25_MODEL = "Qdrant/bm25"
BM25_LANGUAGE = "russian"
# Локальная копия модели в <корень проекта>/models/bm25; если её нет, fastembed скачает модель сам
BM25_LOCAL_PATH = Path(__file__).resolve().parents[2] / "models" / "bm25"


def make_bm25(avg_len: float = 256.0) -> SparseTextEmbedding:
    local = str(BM25_LOCAL_PATH) if BM25_LOCAL_PATH.exists() else None
    return SparseTextEmbedding(
        BM25_MODEL, language=BM25_LANGUAGE, avg_len=avg_len, specific_model_path=local,
    )