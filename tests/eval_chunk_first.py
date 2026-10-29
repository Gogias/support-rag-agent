"""Оценка поиска на наборе chunk_first: правильный абзац выбран до поиска, без pool bias.

Запуск: python tests/eval_chunk_first.py
Берёт tests/golden_sets/chunk_first.json (проверенный вручную), а если его ещё нет —
черновик chunk_first_draft.json. Черновик не проверен: B2B-вопросы отсеиваются по папке
и заголовку, двойники ответа не отмечены, поэтому цифра по черновику занижена.
Отчёт: tests/results/eval_results/chunk_first_<дата-время>.txt
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from eval_retrieval import KS, first_answer, is_off_segment, run_query, wilson  # noqa: E402

from src.agent.retrieval import RETRIEVE_K  # noqa: E402
from src.ingestion.segment import segment  # noqa: E402

HERE = Path(__file__).parent
FINAL_PATH = HERE / "golden_sets" / "chunk_first.json"
DRAFT_PATH = HERE / "golden_sets" / "chunk_first_draft.json"
RESULTS_DIR = HERE / "results" / "eval_results"
BUSINESS = re.compile(r"бизнес|корпоратив|организаци|юрлиц|юридическ", re.I)
USE_DRAFT = True  # True — весь черновик (не проверен), False — chunk_first.json (проверен вручную)
# версия кода, на которой снят отчёт; -dirty — были незакоммиченные изменения
COMMIT = subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=ROOT, text=True).strip()


def load():
    """Единый формат: id, query, answer (множество chunk_id), tag. Второй элемент — проверен ли набор."""
    if not USE_DRAFT and FINAL_PATH.exists():
        items = [x for x in json.loads(FINAL_PATH.read_text(encoding="utf-8")) if x["keep"]]
        if items:
            return [{"id": x["id"], "query": x["query"], "answer": set(x["answer_chunk_ids"]), "tag": x["tag"]}
                    for x in items], True
    items, dropped = [], 0
    for x in json.loads(DRAFT_PATH.read_text(encoding="utf-8")):
        if not x["question"]:
            continue
        if not any(segment(s) == "b2c" for s in x["sources"]) or BUSINESS.search(x.get("title") or ""):
            dropped += 1
            continue
        items.append({"id": x["cand_id"], "query": x["question"], "answer": set(x.get("chunk_ids") or [x["chunk_id"]]),
                      "tag": "negative" if x["negative"] else "other"})
    print(f"Черновик: отсеяно как B2B {dropped}")
    return items, False


def main():
    items, verified = load()
    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    ranks, off, start = {}, [], time.perf_counter()
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}:", end="", flush=True)
        results, _ = run_query(item["query"], lambda step: print(f" {step}", end="", flush=True))
        for name, payloads in results.items():
            ranks.setdefault(name, []).append(first_answer(payloads, item["answer"]))
        llm_chunks = results["search→LLM"]
        off.append((sum(map(is_off_segment, llm_chunks)), len(llm_chunks)))
        print(f"  — {'попал' if ranks['search→LLM'][-1] else 'промах'}")
    print(f"\nВсего: {time.perf_counter() - start:.0f} с\n")

    n = len(items)
    out(f"commit: {COMMIT}")
    out(f"Набор: {'chunk_first.json (проверен вручную)' if verified else 'chunk_first_draft.json (НЕ проверен, цифра занижена)'}")
    out(f"Вопросов: {n}")
    out()
    out(f"{'этап':<16}" + "".join(f"{'AHit@' + str(k):>9}" for k in KS) + f"{'MRR':>9}")
    for name, rs in ranks.items():
        hits = [sum(r is not None and r <= k for r in rs) / n for k in KS]
        out(f"{name:<16}" + "".join(f"{h:>9.0%}" for h in hits) + f"{sum(1 / r for r in rs if r) / n:>9.3f}")

    k = sum(r is not None for r in ranks["search→LLM"])
    lo, hi = wilson(k, n)
    out(f"\nAnswerHit@LLM = {k}/{n} ({k / n:.0%}), 95% ДИ {lo:.0%}–{hi:.0%}")

    out("\nПо типам вопросов:")
    for tag in sorted({x["tag"] for x in items}):
        idx = [i for i, x in enumerate(items) if x["tag"] == tag]
        hit = sum(ranks["search→LLM"][i] is not None for i in idx)
        out(f"  {tag:<14} {hit}/{len(idx)} ({hit / len(idx):.0%})")

    off_n, total = sum(o for o, _ in off), sum(t for _, t in off)
    out(f"\nOffSegment@LLM: B2B и банковских чанков в контексте {off_n}/{total} ({off_n / total:.0%}), "
        f"запросов хотя бы с одним таким: {sum(o > 0 for o, _ in off)}/{n}")

    out("\nПромахи (где потерян абзац с ответом):")
    for i, item in enumerate(items):
        if ranks["search→LLM"][i] is not None:
            continue
        r = ranks["hybrid+rerank"][i]
        where = f"нет среди {RETRIEVE_K} кандидатов" if r is None else f"позиция {r} после реранкера"
        out(f"  [{item['id']}] {item['tag']:<9} {where}: {item['query']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"chunk_first_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()