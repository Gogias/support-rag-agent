"""
chunk_and_embed.py

- Парсится YAML-frontmatter: title/product/section/slug/updated выносятся
  в payload отдельными полями. В `content` остаётся только тело статьи
  (frontmatter больше не попадает ни в эмбеддинг, ни в выдачу LLM).
- `updated` нормализуется к RFC 3339 (ISO 8601 с Z) — так требует
  Qdrant для `PayloadSchemaType.DATETIME`.
- Создаются payload-индексы (keyword/datetime/text) ДО заливки.
- Bulk upsert с wait=False, финальный батч — wait=True.
- Ретраи на llama-server (3 попытки с экспоненциальным backoff).

# FIX: разбиение теперь трёхуровневое:
#   1) по markdown-заголовкам ##/###/#### (строится heading_path)
#   2) внутри секции — по началу нумерованных списков (как раньше,
#      но без потери пустых строк)
#   3) recursive_split по токенам, если секция всё ещё большая
# FIX: перед чанкингом из текста вычищаются markdown-картинки (шум токенов)
# FIX: при эмбеддинге к тексту чанка приклеивается контекстный префикс
#   (product > section > title > heading_path) — сам "сырой" content
#   в payload при этом остаётся чистым, без префикса
# FIX: в payload добавлены chunk_id/prev_chunk_id/next_chunk_id/heading_path
#   для windowed-retrieval (подтянуть соседний чанк при генерации ответа)
"""

import hashlib
import re
import time
import uuid
import yaml
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from transformers import AutoTokenizer
from tqdm import tqdm

from bm25 import make_bm25


raw_text_folder_name = "raw_text_old"

# ---- Путь до данных ------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / raw_text_folder_name

# ---- Конфиг --------------------------------------------------------------
MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "MTS_support_v2"   # HYBRID: новая коллекция, старую не трогаем
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
VECTOR_SIZE = 1024
MAX_TOKENS = 1024
OVERLAP_TOKENS = 130
QDRANT_URL = "http://localhost:6333"
LLAMA_SERVER_URL = "http://localhost:5002/embedding"
POINT_ID_NAMESPACE = uuid.UUID("f4b8c1a0-0000-4000-8000-000000000000")

BATCH_SIZE = 256
EMBED_RETRIES = 3
EMBED_RETRY_DELAY = 2.0

SEPARATOR_HIERARCHY = ["\n\n", "\n", ". ", " "]

# Токенизатор — только из локального кэша
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


# ---- Frontmatter ---------------------------------------------------------
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Возвращает (metadata, body). Если frontmatter нет — ({}, text)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_str, body = m.groups()
    try:
        metadata = yaml.safe_load(fm_str) or {}
    except yaml.YAMLError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, body.lstrip("\n")


def normalize_datetime(value) -> str | None:
    """'Wed, 18 Feb 2026 07:46:48 GMT' -> '2026-02-18T07:46:48Z'."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        dt = datetime.strptime(str(value), "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


# ---- FIX: чистка markdown-шума перед чанкингом ----------------------------
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def strip_markdown_noise(text: str) -> str:
    """Убирает markdown-картинки (URL/хэши — чистый шум для эмбеддинга).
    Ссылки [текст](url) намеренно не трогаем — анкорный текст полезен."""
    text = IMAGE_RE.sub("", text)
    # схлопываем тройные+ переводы строк, оставшиеся после вырезания картинок
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---- FIX: Уровень 0: разбиение по markdown-заголовкам ---------------------
HEADER_LINE_RE = re.compile(r"^(#{2,4})\s+(.*)$")


def split_by_headers(text: str) -> list[tuple[list[str], str]]:
    """Возвращает список (heading_path, section_text).
    heading_path — иерархия заголовков ('##','###','####') над секцией.
    Если заголовков нет вообще — одна секция с пустым heading_path."""
    lines = text.split("\n")
    sections: list[tuple[list[str], list[str]]] = []
    path_stack: list[tuple[int, str]] = []
    current: list[str] = []

    def flush():
        if current and "\n".join(current).strip():
            sections.append(([t for _, t in path_stack], current[:]))

    for line in lines:
        m = HEADER_LINE_RE.match(line)
        if m:
            flush()
            current = []
            level = len(m.group(1))
            title = m.group(2).strip(" *")
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))
        current.append(line)
    flush()

    if not sections:
        return [([], text)]
    return [(path, "\n".join(lns)) for path, lns in sections]


# ---- Уровень 1: разбиение секции на структурные подсекции -----------------
NUMBERED_ITEM_RE = re.compile(r"^\s*1[.)]\s")


def split_into_subsections(text: str) -> list[str]:
    """FIX: пустые строки больше не выбрасываются — они нужны как
    разделитель '\\n\\n' на следующем уровне (recursive_split)."""
    lines = text.split("\n")
    non_empty_idx = [i for i, l in enumerate(lines) if l.strip()]

    boundaries = []
    for j in range(len(non_empty_idx) - 1):
        i, nxt = non_empty_idx[j], non_empty_idx[j + 1]
        if NUMBERED_ITEM_RE.match(lines[nxt]) and not NUMBERED_ITEM_RE.match(lines[i]):
            boundaries.append(nxt)

    if not boundaries:
        joined = "\n".join(lines).strip()
        return [joined] if joined else []

    subsections, start = [], 0
    for b in boundaries:
        if b > start:
            chunk = "\n".join(lines[start:b]).strip()
            if chunk:
                subsections.append(chunk)
        start = b
    tail = "\n".join(lines[start:]).strip()
    if tail:
        subsections.append(tail)
    return subsections


# ---- Уровень 2: рекурсивное досекание --------------------------------------
def recursive_split(text: str, max_tokens: int, separators: list[str] | None = None) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    separators = SEPARATOR_HIERARCHY if separators is None else separators
    if not separators:
        token_ids = _tokenizer.encode(text, add_special_tokens=False)
        return [
            _tokenizer.decode(token_ids[i:i + max_tokens])
            for i in range(0, len(token_ids), max_tokens)
        ]

    sep, rest_separators = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return recursive_split(text, max_tokens, rest_separators)

    chunks, current, current_tokens = [], [], 0
    for part in parts:
        part_tokens = count_tokens(part)
        if part_tokens > max_tokens:
            if current:
                chunks.append(sep.join(current))
                current, current_tokens = [], 0
            chunks.extend(recursive_split(part, max_tokens, rest_separators))
            continue
        if current and current_tokens + part_tokens > max_tokens:
            chunks.append(sep.join(current))
            current, current_tokens = [], 0
        current.append(part)
        current_tokens += part_tokens

    if current:
        chunks.append(sep.join(current))
    return chunks


def add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_ids = _tokenizer.encode(chunks[i - 1], add_special_tokens=False)
        tail_ids = prev_ids[-overlap_tokens:]
        tail_text = _tokenizer.decode(tail_ids)
        result.append(f"{tail_text}\n{chunks[i]}")
    return result


def file_to_chunks(body: str, metadata: dict, source_path: str) -> list[dict]:
    """Возвращает список чанков с готовым payload для Qdrant."""
    base_payload = {
        "title":       metadata.get("title"),
        "product":     metadata.get("product"),
        "section":     metadata.get("section"),
        "slug":        metadata.get("slug"),
        "updated":     normalize_datetime(metadata.get("updated")),
        "source_path": source_path,
    }

    body = strip_markdown_noise(body)  # FIX

    # (heading_path, text) для каждого итогового чанка, до overlap
    pieces: list[tuple[list[str], str]] = []

    for heading_path, section_text in split_by_headers(body):  # FIX
        section_text = section_text.strip()
        if not section_text:
            continue

        if count_tokens(section_text) <= MAX_TOKENS:
            pieces.append((heading_path, section_text))
            continue

        for sub in split_into_subsections(section_text):
            if count_tokens(sub) <= MAX_TOKENS:
                pieces.append((heading_path, sub))
                continue
            sub_parts = add_overlap(recursive_split(sub, MAX_TOKENS), OVERLAP_TOKENS)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    pieces.append((heading_path, sp))

    total = len(pieces)
    chunks = []
    for i, (heading_path, text) in enumerate(pieces):
        chunk_id = f"{source_path}::{i:04d}"
        chunks.append({
            **base_payload,
            "content":        text,
            "chunk_index":    i,
            "total_chunks":   total,
            "chunk_id":       chunk_id,                                            # FIX
            "prev_chunk_id":  f"{source_path}::{i-1:04d}" if i > 0 else None,       # FIX
            "next_chunk_id":  f"{source_path}::{i+1:04d}" if i < total - 1 else None,  # FIX
            "heading_path":   heading_path,                                        # FIX
        })
    return chunks


# ---- DEDUP ----------------------------------------------------------------
def content_hash(text: str) -> str:
    """Хэш нормализованного текста: регистр и пробелы не влияют."""
    norm = " ".join(text.split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Оставляет первую копию каждого текста. Откуда ещё пришли копии,
    сохраняется в списках sources/products/sections/chunk_ids:
    - фильтр по product/section ищет по спискам (MatchValue по массиву
      срабатывает, если совпал любой элемент);
    - lookup соседей идёт по chunk_ids, поэтому prev/next_chunk_id
      выброшенной копии всё равно находит оставшуюся точку."""
    kept: dict[str, dict] = {}
    for c in chunks:
        h = content_hash(c["content"])
        k = kept.get(h)
        if k is None:
            c["content_hash"] = h
            c["sources"] = [c["source_path"]]
            c["products"] = [c["product"]] if c["product"] else []
            c["sections"] = [c["section"]] if c["section"] else []
            c["chunk_ids"] = [c["chunk_id"]]
            kept[h] = c
            continue
        k["sources"].append(c["source_path"])
        k["chunk_ids"].append(c["chunk_id"])
        if c["product"] and c["product"] not in k["products"]:
            k["products"].append(c["product"])
        if c["section"] and c["section"] not in k["sections"]:
            k["sections"].append(c["section"])
    return list(kept.values())


def make_point_id(source_path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{source_path}::{chunk_index:04d}"))


# ---- FIX: контекстный префикс для эмбеддинга (не хранится в content) ------
def build_embed_text(chunk: dict) -> str:
    """Contextual retrieval: приклеиваем к тексту чанка иерархию
    product > section > title > heading_path, чтобы мелкий чанк без
    антецедентов ('эта опция', 'он') всё равно был правильно эмбеднут.
    В payload['content'] префикс НЕ попадает — LLM получает чистый текст."""
    bits = [b for b in (chunk.get("product"), chunk.get("section"), chunk.get("title")) if b]
    bits.extend(chunk.get("heading_path") or [])
    header = " > ".join(bits)
    return f"{header}\n\n{chunk['content']}" if header else chunk["content"]


# ---- Обход папки -----------------------------------------------------------
def collect_md_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"raw_text не существует: {root}")
    files = sorted(f for f in root.rglob("*.md") if f.name.lower() != "index.md")
    if not files:
        raise FileNotFoundError(f"В {root} не найдено .md файлов (кроме index.md)")
    return files


def build_source_path(md_path: Path, root: Path) -> str:
    return md_path.relative_to(root).with_suffix("").as_posix()


# ---- Qdrant ------------------------------------------------------------------
PAYLOAD_INDEXES = [
    ("product",     PayloadSchemaType.KEYWORD),
    ("section",     PayloadSchemaType.KEYWORD),
    ("slug",        PayloadSchemaType.KEYWORD),
    ("source_path", PayloadSchemaType.KEYWORD),
    ("chunk_id",    PayloadSchemaType.KEYWORD),   # FIX: индекс для lookup соседей
    ("chunk_ids",   PayloadSchemaType.KEYWORD),   # DEDUP: все chunk_id, слитые в точку
    ("products",    PayloadSchemaType.KEYWORD),   # DEDUP
    ("sections",    PayloadSchemaType.KEYWORD),   # DEDUP
    ("content_hash", PayloadSchemaType.KEYWORD),  # DEDUP
    ("updated",     PayloadSchemaType.DATETIME),
    ("title",       PayloadSchemaType.TEXT),
]


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_NAME: VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True,   # экономия RAM при росте корпуса
                ),
            },
            # HYBRID: IDF считает Qdrant по всей коллекции — без него это не BM25
            sparse_vectors_config={
                SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
            shard_number=1,
        )
        print(f"Создана коллекция '{COLLECTION_NAME}'")
    else:
        info = client.get_collection(COLLECTION_NAME)
        if not info.config.params.sparse_vectors or SPARSE_NAME not in info.config.params.sparse_vectors:
            raise RuntimeError(
                f"Коллекция '{COLLECTION_NAME}' существует без sparse-вектора '{SPARSE_NAME}'. "
                "Удалите её или смените COLLECTION_NAME."
            )

    for field, schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
        except Exception:
            pass  # индекс уже существует


# ---- Эмбеддинг -----------------------------------------------------------
def embed_text(text: str) -> list[float] | None:
    last_err = None
    for attempt in range(EMBED_RETRIES):
        try:
            resp = requests.post(
                LLAMA_SERVER_URL,
                json={"content": text},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = np.asarray(data[0]["embedding"][0], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.tolist()
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            if attempt < EMBED_RETRIES - 1:
                time.sleep(EMBED_RETRY_DELAY * (attempt + 1))
    print(f"  [embed fail] {last_err}")
    return None


def embed_and_upload(client: QdrantClient, chunks: list[dict], bm25):
    skipped: list[str] = []
    total = len(chunks)

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding + uploading"):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [build_embed_text(c) for c in batch]   # тот же текст для dense и sparse
        sparse_batch = list(bm25.embed(texts))         # HYBRID: embed() для документов
        points: list[PointStruct] = []

        for chunk, text, sp in zip(batch, texts, sparse_batch):
            emb = embed_text(text)  # FIX: эмбеддим с контекстным префиксом
            if emb is None:
                skipped.append(f"{chunk['source_path']}::chunk {chunk['chunk_index']}")
                continue
            vector = {DENSE_NAME: emb}
            if len(sp.indices) > 0:  # чанк только из стоп-слов — sparse не пишем
                vector[SPARSE_NAME] = SparseVector(
                    indices=sp.indices.tolist(),
                    values=sp.values.tolist(),
                )
            points.append(PointStruct(
                id=make_point_id(chunk["source_path"], chunk["chunk_index"]),
                vector=vector,
                payload=chunk,
            ))

        if not points:
            continue

        is_last = (i + BATCH_SIZE) >= total
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=is_last,
        )

    if skipped:
        print(f"\nПропущено чанков: {len(skipped)}")
        for s in skipped:
            print(f"  {s}")


# ---- main ------------------------------------------------------------------
def main():
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)

    md_files = collect_md_files(DATA_ROOT)
    print(f"Найдено файлов: {len(md_files)}")

    all_chunks: list[dict] = []
    for md_path in tqdm(md_files, desc="Чанкинг"):
        raw = md_path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw)
        source_path = build_source_path(md_path, DATA_ROOT)
        all_chunks.extend(file_to_chunks(body, metadata, source_path))

    before = len(all_chunks)
    all_chunks = dedup_chunks(all_chunks)  # DEDUP
    print(f"Всего чанков: {before}, после дедупликации: {len(all_chunks)}")

    # HYBRID: avg_len по всему корпусу (в словах — приближённо, для BM25 этого достаточно)
    avg_len = sum(len(build_embed_text(c).split()) for c in all_chunks) / len(all_chunks)
    print(f"BM25 avg_len ≈ {avg_len:.0f}")
    bm25 = make_bm25(avg_len)

    embed_and_upload(client, all_chunks, bm25)

    final_count = client.count(COLLECTION_NAME, exact=True).count
    print(f"Готово. В коллекции '{COLLECTION_NAME}': {final_count} точек.")


if __name__ == "__main__":
    main()