"""
match_golden_set.py — сопоставляет вопросы golden-сета со статьями в MTS_support_v2.

  1. hybrid-поиск (dense + BM25, RRF) + доп. ветка с фильтром products/sections
  2. реранкер (bge-reranker-v2-m3) -> top RERANK_K
  3. соседи по prev/next_chunk_id (через индекс chunk_ids)
  4. LLM-судья по каждому из top RERANK_K: GigaChat API (по умолчанию)
     или локальная модель через KoboldCpp — переключается JUDGE_BACKEND
  5. в запись пишутся relevant_sources (статьи — переживают перечанкинг),
     relevant_chunk_ids и relevant_ids (point ID — для текущей коллекции)

Прогон возобновляемый: результат сохраняется после каждой записи, повторный
запуск продолжает с места остановки (обрабатываются только записи без
match_method или с match_method="error").
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import models

from hybrid_search import COLLECTION_NAME, client, hybrid_search

# ──────────────────────────────────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = SCRIPT_DIR / "golden_sets" / "golden_set_MTS_base.json"
OUTPUT_PATH = SCRIPT_DIR / "golden_sets" / "golden_set_matched.json"
REVIEW_PATH = SCRIPT_DIR / "golden_sets" / "review_needed.jsonl"

CANDIDATE_K = 50          # hybrid без фильтра
FILTER_CANDIDATE_K = 30   # hybrid с фильтром products/sections
RERANK_K = 10             # сколько уходит судье
RERANK_MAX_CHARS = 2000   # обрезка пассажа для реранкера (скорость)
NEIGHBOR_MAX_CHARS = 800  # обрезка соседних чанков в промпте судьи

# llama-server с --reranking (bge-reranker-v2-m3 GGUF), см. команду запуска в чате
RERANKER_URL = os.getenv("RERANKER_URL", "http://127.0.0.1:5003/v1/rerank")

JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "gigachat")  # "gigachat" | "local"
JUDGE_MAX_TOKENS = 300    # потолок генерации: судья не может "зациклиться"
JUDGE_TIMEOUT = 120       # секунд на один вызов судьи
JUDGE_RETRIES = 4         # повторы при 429 (один поток на физлицо) и 5xx
REQUIRE_HIGH_CONFIDENCE = False  # True — засчитывать только "ДА + высокая"

# GigaChat: ключ авторизации и scope — из .env (см. объяснение в чате)
GIGACHAT_BASE_URL = "https://api.giga.chat/v1"
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra")

# Локальный судья (запасной вариант)
KOBOLD_URL = "http://localhost:5001/v1"
LOCAL_MODEL = "qwen3.5"

load_dotenv(SCRIPT_DIR.parent.parent / ".env")
from langfuse import get_client, observe, propagate_attributes  # noqa: E402

langfuse = get_client()
token_usage = {"prompt": 0, "completion": 0}

if JUDGE_BACKEND == "gigachat":
    from gigachat import GigaChat  # noqa: E402
    from gigachat.models import Chat, Messages, MessagesRole  # noqa: E402

    # credentials/scope/сертификаты SDK сам читает из переменных GIGACHAT_*;
    # access-токен (живёт 30 мин) SDK получает и обновляет автоматически
    giga = GigaChat(
        base_url=GIGACHAT_BASE_URL,
        model=GIGACHAT_MODEL,
        timeout=JUDGE_TIMEOUT,
        max_retries=JUDGE_RETRIES,      # повторы при 429/5xx делает сам SDK
        retry_backoff_factor=2.0,       # паузы растут: 2, 4, 8... с
    )
    JUDGE_MODEL = GIGACHAT_MODEL
else:
    from langfuse.openai import openai  # noqa: E402

    llm = openai.OpenAI(base_url=KOBOLD_URL, api_key="not-needed",
                        timeout=JUDGE_TIMEOUT, max_retries=1)
    JUDGE_MODEL = LOCAL_MODEL

JUDGE_SYSTEM_PROMPT = (
    "Ты — строгий эксперт-разметчик, оценивающий качество поиска в RAG-системе "
    "техподдержки МТС. Оцени, отвечает ли фрагмент документации на вопрос "
    "пользователя. Не додумывай: если ответа нет прямо во фрагменте — так и скажи.\n"
    "Отвечай сразу, без рассуждений, строго в требуемом формате, на русском языке."
)

JUDGE_PROMPT = """Вопрос пользователя: "{query}"

Фрагмент документации (статья: «{title}», раздел: «{heading}»):
---
{context}
---

Отвечает ли ОСНОВНОЙ ЧАНК прямо и по существу на вопрос пользователя? \
Соседние чанки — только для контекста.

Ответь строго в формате:
ВЕРДИКТ: ДА или НЕТ
УВЕРЕННОСТЬ: высокая или низкая
"""

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


# ──────────────────────────────────────────────────────────────────────────
# Данные
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    point_id: str
    payload: dict
    fused_score: float
    rerank_score: float | None = None
    context: str = ""
    verdict: bool | None = None
    high_confidence: bool | None = None
    judge_raw: str = ""


@dataclass
class MatchOutcome:
    candidates: list[Candidate] = field(default_factory=list)
    confirmed: list[Candidate] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# 1. Поиск
# ──────────────────────────────────────────────────────────────────────────

def build_filter(product: str | None, section: str | None) -> models.Filter | None:
    conditions = []
    if product:
        conditions.append(models.FieldCondition(key="products", match=models.MatchValue(value=product)))
    if section:
        conditions.append(models.FieldCondition(key="sections", match=models.MatchValue(value=section)))
    return models.Filter(must=conditions) if conditions else None


def search_candidates(query: str, product: str | None, section: str | None) -> list[Candidate]:
    branches = [(None, CANDIDATE_K)]
    query_filter = build_filter(product, section)
    if query_filter:
        branches.append((query_filter, FILTER_CANDIDATE_K))

    by_id: dict[str, Candidate] = {}
    for f, limit in branches:
        for p in hybrid_search(query, limit=limit, query_filter=f):
            c = by_id.get(p.id)
            if c is None or p.score > c.fused_score:
                by_id[p.id] = Candidate(point_id=p.id, payload=p.payload, fused_score=p.score)
    return list(by_id.values())


# ──────────────────────────────────────────────────────────────────────────
# 2. Реранкер
# ──────────────────────────────────────────────────────────────────────────

def rerank_via_service(query: str, passages: list[str]) -> list[float]:
    # формат llama-server: {"query", "documents"} -> {"results": [{"index", "relevance_score"}]}
    resp = requests.post(RERANKER_URL, json={"query": query, "documents": passages}, timeout=120)
    resp.raise_for_status()
    scores = {r["index"]: r["relevance_score"] for r in resp.json()["results"]}
    return [scores[i] for i in range(len(passages))]


def rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    passages = []
    for c in candidates:
        heading = " > ".join(c.payload.get("heading_path") or [])
        text = f"{c.payload.get('title', '')}\n{heading}\n{c.payload.get('content', '')}"
        passages.append(text[:RERANK_MAX_CHARS])
    for c, s in zip(candidates, rerank_via_service(query, passages)):
        c.rerank_score = float(s)
    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:RERANK_K]


# ──────────────────────────────────────────────────────────────────────────
# 3. Соседи — по chunk_ids (индексированное поле, переживает дедупликацию)
# ──────────────────────────────────────────────────────────────────────────

def add_context(c: Candidate) -> None:
    prev_id, next_id = c.payload.get("prev_chunk_id"), c.payload.get("next_chunk_id")
    wanted = [x for x in (prev_id, next_id) if x]
    neighbors = {}
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
                    neighbors[cid] = p.payload["content"][:NEIGHBOR_MAX_CHARS]

    blocks = []
    if prev_id in neighbors:
        blocks.append(f"[СОСЕДНИЙ ЧАНК ДО]\n{neighbors[prev_id]}")
    blocks.append(f"[ОСНОВНОЙ ЧАНК]\n{c.payload.get('content', '')}")
    if next_id in neighbors:
        blocks.append(f"[СОСЕДНИЙ ЧАНК ПОСЛЕ]\n{neighbors[next_id]}")
    c.context = "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────
# 4. Судья
# ──────────────────────────────────────────────────────────────────────────

@observe(name="judge_gigachat", as_type="generation")
def ask_gigachat(prompt: str) -> str:
    chat = Chat(
        messages=[Messages(role=MessagesRole.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
                  Messages(role=MessagesRole.USER, content=prompt)],
        temperature=0.01,  # почти детерминированно
        max_tokens=JUDGE_MAX_TOKENS,
    )
    resp = giga.chat(chat)
    usage = resp.usage
    token_usage["prompt"] += usage.prompt_tokens
    token_usage["completion"] += usage.completion_tokens
    langfuse.update_current_generation(
        model=GIGACHAT_MODEL,
        usage_details={"input": usage.prompt_tokens, "output": usage.completion_tokens},
    )
    return resp.choices[0].message.content or ""


def ask_local(prompt: str) -> str:
    resp = llm.chat.completions.create(
        model=LOCAL_MODEL,
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        # выключаем reasoning: discard_think только прячет рассуждение,
        # а модель всё равно тратит на него время
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


def judge(query: str, c: Candidate) -> None:
    prompt = JUDGE_PROMPT.format(
        query=query,
        title=c.payload.get("title", ""),
        heading=" > ".join(c.payload.get("heading_path") or []),
        context=c.context,
    )
    raw = ask_gigachat(prompt) if JUDGE_BACKEND == "gigachat" else ask_local(prompt)
    raw = THINK_RE.sub("", raw).strip()
    c.judge_raw = raw

    verdict = next((l for l in raw.splitlines() if "ВЕРДИКТ" in l.upper()), "").upper()
    conf = next((l for l in raw.splitlines() if "УВЕРЕННОСТЬ" in l.upper()), "").lower()
    c.verdict = "ДА" in verdict and "НЕТ" not in verdict  # нет строки ВЕРДИКТ -> НЕТ
    c.high_confidence = "высок" in conf


def is_confirmed(c: Candidate) -> bool:
    return bool(c.verdict) and (c.high_confidence or not REQUIRE_HIGH_CONFIDENCE)


# ──────────────────────────────────────────────────────────────────────────
# Одна запись
# ──────────────────────────────────────────────────────────────────────────

@observe(name="match_golden_set_entry")
def match_entry(entry: dict) -> MatchOutcome:
    with propagate_attributes(metadata={"entry_id": str(entry.get("id"))},
                              tags=["golden-set-matching"]):
        query = entry["query"]
        top = rerank(query, search_candidates(query, entry.get("product"), entry.get("section")))
        for c in top:
            add_context(c)
            judge(query, c)

        outcome = MatchOutcome(candidates=top, confirmed=[c for c in top if is_confirmed(c)])
        if outcome.confirmed:
            sources = {s for c in outcome.confirmed for s in c.payload.get("sources", [])}
            entry["relevant_sources"] = sorted(sources)
            entry["relevant_chunk_ids"] = [c.payload["chunk_id"] for c in outcome.confirmed]
            entry["relevant_ids"] = [c.point_id for c in outcome.confirmed]
            entry["verified"] = True
            entry["match_method"] = "auto_llm_judge"
            entry["judge_model"] = JUDGE_MODEL
        else:
            entry["relevant_sources"] = entry["relevant_chunk_ids"] = entry["relevant_ids"] = None
            entry["verified"] = False
            entry["match_method"] = "needs_manual_review"
        entry.pop("error", None)
        return outcome


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────

def load_entries() -> list[dict]:
    path = OUTPUT_PATH if OUTPUT_PATH.exists() else GOLDEN_SET_PATH  # возобновление
    print(f"Читаю {path}")
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def save_entries(entries: list[dict]) -> None:
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUTPUT_PATH)  # атомарно: прерывание не портит файл


def review_row(entry: dict, outcome: MatchOutcome | None) -> dict:
    return {
        "id": entry.get("id"),
        "query": entry["query"],
        "error": entry.get("error"),
        "candidates": [{
            "point_id": c.point_id,
            "rerank_score": round(c.rerank_score, 4),
            "title": c.payload.get("title"),
            "heading_path": c.payload.get("heading_path"),
            "sources": c.payload.get("sources"),
            "verdict": c.verdict,
            "high_confidence": c.high_confidence,
            "judge_raw": c.judge_raw,
            "content_preview": c.payload.get("content", "")[:300],
        } for c in (outcome.candidates if outcome else [])],
    }


def preflight() -> None:
    """Проверяет все сервисы ДО прогона: лучше упасть сразу, чем пометить
    ошибкой все записи подряд."""
    checks = {
        "Qdrant": lambda: client.get_collection(COLLECTION_NAME),
        "Эмбеддинги (llama-server :5002)": lambda: hybrid_search("проверка", limit=1),
        "Реранкер": lambda: rerank_via_service("проверка", ["тест"]),
        "Судья": lambda: (ask_gigachat if JUDGE_BACKEND == "gigachat" else ask_local)("Ответь: ОК"),
    }
    failed = False
    for name, check in checks.items():
        try:
            check()
            print(f"  ✓ {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name}: {exc}", file=sys.stderr)
            failed = True
    if failed:
        sys.exit("Не все сервисы доступны — прогон не запущен.")


def main():
    print(f"Судья: {JUDGE_BACKEND} / {JUDGE_MODEL}")
    preflight()
    entries = load_entries()
    todo = [e for e in entries if e.get("match_method") in (None, "error")]
    print(f"Записей: {len(entries)}, к обработке: {len(todo)}")

    with REVIEW_PATH.open("a", encoding="utf-8") as review:
        for i, entry in enumerate(todo, 1):
            try:
                outcome = match_entry(entry)
            except Exception as exc:  # noqa: BLE001
                entry["match_method"], entry["error"] = "error", str(exc)
                outcome = None
                print(f"[{i}/{len(todo)}] ОШИБКА: {exc}", file=sys.stderr)

            if entry["match_method"] != "auto_llm_judge":
                review.write(json.dumps(review_row(entry, outcome), ensure_ascii=False) + "\n")
                review.flush()
            else:
                print(f"[{i}/{len(todo)}] ✓ {len(entry['relevant_sources'])} ст. — {entry['query'][:60]}")
            if entry["match_method"] == "needs_manual_review":
                print(f"[{i}/{len(todo)}] ? проверка — {entry['query'][:60]}")
            save_entries(entries)

    langfuse.flush()
    stats = {}
    for e in entries:
        stats[e.get("match_method")] = stats.get(e.get("match_method"), 0) + 1
    print(f"\nИтог: {stats}")
    if JUDGE_BACKEND == "gigachat":
        print(f"Токены GigaChat за прогон: промпт {token_usage['prompt']:,}, "
              f"ответ {token_usage['completion']:,}")


if __name__ == "__main__":
    main()