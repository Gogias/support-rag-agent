"""Проверка LLM-судьи на случайной выборке чанков (слепая ручная оценка).

Запуск: python tests/review_golden.py

Берёт из golden set v4 по PER_GRADE чанков на каждую оценку судьи:
  1–3 — из judgments; 0 — из dense-топ-5 (был в пуле судьи), которых нет в judgments.
Показывает запрос и чанк, вы ставите оценку 0–3, не видя оценки судьи.
Результат: tests/golden_sets/judge_review_sample.json, сохраняется после каждой оценки;
q — выход, повторный запуск продолжит. Golden set не меняется.
"""
import json
import random
import sys
from pathlib import Path

from qdrant_client import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.retrieval import COLLECTION_NAME, client, embed_query  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "golden_sets" / "golden_set_MTS_v4.json"
OUT_PATH = Path(__file__).parent / "golden_sets" / "judge_review_sample.json"
PER_GRADE = 25   # чанков на каждую оценку судьи 0, 1, 2, 3 -> всего до 100
TOP_K = 5        # dense-топ, из которого берутся чанки с оценкой 0 (в relabel_golden он был в пуле)
SEED = 42


def chunk_text(chunk_id):
    points, _ = client.scroll(
        COLLECTION_NAME,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="chunk_ids", match=models.MatchAny(any=[chunk_id])),
        ]),
        limit=1, with_payload=["title", "heading_path", "content"],
    )
    if not points:
        return "(чанк не найден в коллекции)"
    p = points[0].payload
    path = " > ".join([p.get("title") or "", *(p.get("heading_path") or [])])
    return f"{path}\n{p.get('content', '')}"


def build_sample(golden, rng):
    by_grade = {g: [] for g in range(4)}
    for item in golden:
        for j in item.get("judgments", []):
            by_grade[j["grade"]].append({"id": item["id"], "query": item["query"],
                                         "chunk_id": j["chunk_id"], "judge_grade": j["grade"]})

    # Оценка 0: dense-топ, который судья видел, но не отметил (в judgments только оценки > 0).
    queries = golden[:]
    rng.shuffle(queries)
    for item in queries:
        if len(by_grade[0]) >= PER_GRADE:
            break
        known = {j["chunk_id"] for j in item.get("judgments", [])}
        top = client.query_points(COLLECTION_NAME, query=embed_query(item["query"]), using="dense",
                                  limit=TOP_K, with_payload=["chunk_id"]).points
        zeros = [p.payload["chunk_id"] for p in top if p.payload.get("chunk_id") not in known]
        if zeros:
            by_grade[0].append({"id": item["id"], "query": item["query"],
                                "chunk_id": rng.choice(zeros), "judge_grade": 0})

    sample = []
    for g, chunks in by_grade.items():
        sample += rng.sample(chunks, min(PER_GRADE, len(chunks)))
    rng.shuffle(sample)  # вперемешку, чтобы порядок не подсказывал оценку
    return sample


def main():
    rng = random.Random(SEED)
    if OUT_PATH.exists():
        sample = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        print("Собираю выборку (для оценки 0 нужен реранкер)...")
        sample = build_sample(golden, rng)
        OUT_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    todo = [s for s in sample if "human_grade" not in s]
    print(f"Выборка: {len(sample)}, осталось оценить: {len(todo)}")
    print("Шкала: 3 — полный ответ, 2 — частичный, 1 — по теме без ответа, 0 — не по теме\n")

    for n, s in enumerate(todo, len(sample) - len(todo) + 1):
        print("═" * 80)
        print(f"{n}/{len(sample)}  ВОПРОС: {s['query']}\n")
        print(chunk_text(s["chunk_id"]))
        while (answer := input("\nОценка 0–3 (q — выход): ").strip().lower()) not in {"0", "1", "2", "3", "q"}:
            pass
        if answer == "q":
            break
        s["human_grade"] = int(answer)
        print(f"  судья: {s['judge_grade']}")
        OUT_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nОценено {sum('human_grade' in s for s in sample)}/{len(sample)}. Файл: {OUT_PATH}")


if __name__ == "__main__":
    main()