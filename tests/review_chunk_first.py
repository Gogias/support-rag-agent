"""chunk_first, этапы 3–4: ручная проверка вопросов с подсказкой двойников.

Запуск: python tests/review_chunk_first.py
Вход:  tests/golden_sets/chunk_first_draft.json (этап 2, gen_questions.py)
Выход: tests/golden_sets/chunk_first.json — сохраняется после каждого вопроса,
       повторный запуск продолжит с места остановки.

Двойники — чанки, ближайшие по вектору к самому абзацу с ответом (не к вопросу),
поэтому поиск по запросу на выбор правильного ответа не влияет.
"""
import json
import sys
from pathlib import Path

from qdrant_client import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.retrieval import COLLECTION_NAME, client  # noqa: E402

HERE = Path(__file__).parent
SRC_PATH = HERE / "golden_sets" / "chunk_first_draft.json"
OUT_PATH = HERE / "golden_sets" / "chunk_first.json"
N_TWINS = 5
TAGS = {"p": "plain", "n": "negative", "v": "vocab", "s": "segment_trap"}
TARGET = 60


def point_id(chunk_id):
    points, _ = client.scroll(COLLECTION_NAME, limit=1, scroll_filter=models.Filter(must=[
        models.FieldCondition(key="chunk_ids", match=models.MatchAny(any=[chunk_id]))]))
    return points[0].id


def twins(chunk_id):
    pid = point_id(chunk_id)
    points = client.query_points(COLLECTION_NAME, query=pid, using="dense", limit=N_TWINS + 1,
                                 with_payload=["chunk_id", "chunk_ids", "title", "heading_path", "content"]).points
    return [p for p in points if p.id != pid][:N_TWINS]


def ask(prompt, allowed=None):
    while True:
        answer = input(prompt).strip()
        if allowed is None or answer in allowed:
            return answer


def main():
    draft = [x for x in json.loads(SRC_PATH.read_text(encoding="utf-8")) if x["question"]]
    result = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else []
    seen = {x["cand_id"] for x in result}

    for c in draft:
        if c["cand_id"] in seen:
            continue
        kept = sum(x["keep"] for x in result)
        if kept >= TARGET:
            print(f"Набрано {kept} вопросов — готово.")
            break

        tw = twins(c["chunk_id"])
        print("\n" + "═" * 100)
        print(f"{c['cand_id']}   оставлено {kept}/{TARGET}   {'[есть отрицание]' if c['negative'] else ''}")
        print(f"Статья: {c.get('title')} > {' > '.join(c.get('heading_path') or [])}\n")
        print(c["content"][:1500])
        print(f"\nВОПРОС: {c['question']}\nОТВЕТ:  {c['answer']}")
        print("\nПохожие абзацы:")
        for i, p in enumerate(tw, 1):
            path = " > ".join([p.payload.get("title") or "", *(p.payload.get("heading_path") or [])])
            print(f"  [{i}] {p.score:.2f}  {path}\n      {p.payload.get('content', '')[:300].replace(chr(10), ' ')}")

        action = ask("\nОставить? y — да, n — нет, e — переписать вопрос, q — выход: ", {"y", "n", "e", "q"})
        if action == "q":
            break
        if action == "n":
            result.append({"cand_id": c["cand_id"], "keep": False})
        else:
            question = input("Новый вопрос: ").strip() if action == "e" else c["question"]
            tag = ask(f"Тег: p plain, n negative, v vocab, s segment_trap [{'n' if c['negative'] else 'p'}]: ",
                      {"", *TAGS}) or ("n" if c["negative"] else "p")
            picked = ask("Номера похожих абзацев, где тоже есть ответ (через пробел, Enter — нет): ")
            ids = set(c.get("chunk_ids") or [c["chunk_id"]])
            for n in picked.split():
                if n.isdigit() and 1 <= int(n) <= len(tw):
                    p = tw[int(n) - 1].payload
                    ids.update(p.get("chunk_ids") or [p["chunk_id"]])
            result.append({
                "cand_id": c["cand_id"], "keep": True,
                "id": f"cf-{sum(x['keep'] for x in result) + 1:03d}",
                "query": question, "tag": TAGS[tag],
                "answer_chunk_ids": sorted(ids), "source_chunk": c["chunk_id"],
                "sources": c["sources"], "reference_answer": c["answer"],
                "answerable": True, "verified": True,
            })
        OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = [x for x in result if x["keep"]]
    print(f"\nПроверено {len(result)}, оставлено {len(kept)}. Сохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
