"""Оценка поиска графа по golden set: метрики по документам и по абзацу с ответом.

Запуск: python tests/eval_retrieval.py
Все этапы берутся из src/agent/retrieval.py — того же модуля, что вызывает узел графа:
  dense, sparse, hybrid (RRF)   — кандидаты до реранкера (Hit@50 = попал ли ответ в пул реранкера)
  dense+rerank, hybrid+rerank   — порядок после реранкера (title + heading_path + content, как в графе)
  search→LLM                    — итог поиска: ровно те чанки, что search() отдаёт в LLM (top-RERANK_K после порога)
Hit@k — есть ли в top-k чанк из релевантного документа (relevant_ids, через sources). Завышена:
  засчитывает любой абзац статьи. Оставлена для сравнения со старыми прогонами.
AnswerHit@k — есть ли в top-k сам чанк с оценкой 3 из judgments, и только из b2c-статьи.
OffSegment@LLM — доля B2B- и банковских чанков в том, что уходит в LLM.
nDCG@5 — учитывает оценки 0–3 из поля relevance (golden set v4) и порядок выдачи.
Запросы с "answerable": false (ответа в базе нет) в метрики не входят — по ним и по скорам
релевантных чанков строится таблица калибровки порога реранкера (RERANK_MIN_SCORE).
Отчёт сохраняется в tests/results/eval_results/<дата-время>.txt.
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from qdrant_client import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.retrieval import (  # noqa: E402
    COLLECTION_NAME, RERANK_K, RERANK_MIN_SCORE, RETRIEVE_K, bm25, client, embed_query, find_candidates, rerank, search,
)
from src.ingestion.segment import segment  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "golden_sets" / "golden_set_MTS_v4.json"
RESULTS_DIR = Path(__file__).parent / "results" / "eval_results"
ONLY_VERIFIED = False  # True — считать только по проверенным вручную запросам
KS = [1, 3, 5, 10, RETRIEVE_K]
NDCG_K = 5
THRESHOLDS = [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2]  # кандидаты порога для таблицы калибровки


def dense(query):
    return [p.payload for p in client.query_points(
        COLLECTION_NAME, query=embed_query(query), using="dense", limit=RETRIEVE_K, with_payload=True
    ).points]


def sparse(query):
    sp = next(iter(bm25.query_embed(query)))
    if not len(sp.indices):
        return []
    return [p.payload for p in client.query_points(
        COLLECTION_NAME, query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
        using="sparse", limit=RETRIEVE_K, with_payload=True,
    ).points]


def docs_of(p):
    return p.get("sources") or [p["source_path"]]


def ids_of(p):
    """Все chunk_id точки: после дедупа одна точка хранит id всех своих копий."""
    return set(p.get("chunk_ids") or [p.get("chunk_id")])


def first_hit(payloads, relevant):
    """Позиция (с 1) первого чанка, документ которого релевантен; None если нет."""
    for i, p in enumerate(payloads, 1):
        if relevant & set(docs_of(p)):
            return i
    return None


def answer_ids(item):
    """Чанки с оценкой 3 из b2c-статей — абзацы с ответом. Чужой сегмент ответом не считается."""
    return {j["chunk_id"] for j in item.get("judgments", [])
            if j["grade"] == 3 and any(segment(s) == "b2c" for s in j["sources"])}


def first_answer(payloads, answer):
    """Позиция (с 1) первого чанка с ответом; None если нет."""
    return next((i for i, p in enumerate(payloads, 1) if ids_of(p) & answer), None)


def is_off_segment(p):
    """Чужой чанк: ни одна из его копий не из b2c-статьи."""
    return all(segment(d) != "b2c" for d in docs_of(p))


def wilson(k, n, z=1.96):
    """95% интервал Уилсона для доли k/n."""
    p = k / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / (1 + z * z / n), (center + spread) / (1 + z * z / n)


def doc_groups(item):
    """Документы, которые лежат в одном чанке (региональные копии и т.п.), — одна группа."""
    group = {}
    for j in item.get("judgments", []):
        roots = {group.get(d, d) for d in j["sources"]}
        root = min(roots)
        for d, g in list(group.items()):
            if g in roots:
                group[d] = root
        for d in j["sources"]:
            group[d] = root
    return group


def ndcg(payloads, relevance, group, k=NDCG_K):
    """nDCG@k по группам документов: каждая группа засчитывается один раз, на первой позиции."""
    seen, dcg = set(), 0.0
    for i, p in enumerate(payloads[:k], 1):
        new = [d for d in docs_of(p) if group.get(d, d) not in seen]
        seen.update(group.get(d, d) for d in docs_of(p))
        grade = max((relevance.get(d, 0) for d in new), default=0)
        dcg += (2 ** grade - 1) / math.log2(i + 1)
    best = {}
    for d, g in relevance.items():
        best[group.get(d, d)] = max(g, best.get(group.get(d, d), 0))
    ideal = sorted(best.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def run_query(query, log):
    """Все этапы для одного запроса; кандидаты считаются один раз и переиспользуются."""
    log("dense")
    d = dense(query)
    log("sparse")
    s = sparse(query)
    log("hybrid")
    h = find_candidates(query)
    log("rerank(dense)")
    dr = [p for _, p in rerank(query, d)]
    log("rerank(hybrid)")
    hr_scored = rerank(query, h)
    log("search")
    g = search(query)
    results = {"dense": d, "sparse": s, "hybrid": h, "dense+rerank": dr,
               "hybrid+rerank": [p for _, p in hr_scored], "search→LLM": g}
    return results, hr_scored


def best_score(scored, relevant=None):
    """Лучший скор реранкера: среди чанков релевантных документов или среди всех."""
    return max((s for s, p in scored if relevant is None or relevant & set(docs_of(p))), default=None)


def main():
    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    golden = [x for x in golden if x["verified"] or not ONLY_VERIFIED]
    items = [x for x in golden if x.get("answerable", True) and x["relevant_ids"]]
    no_answer = [x for x in golden if x.get("answerable") is False]
    skipped = [x["id"] for x in golden if x.get("answerable", True) and not x["relevant_ids"]]
    print(f"{GOLDEN_PATH.name}: с ответом {len(items)}, без ответа {len(no_answer)}, "
          f"на каждый 6 этапов (2 вызова реранкера + search())\n")

    ranks, ndcgs, rel_scores = {}, {}, []
    ans_ranks, ans_items, gaps = {}, [], []
    no_answer_chunk = {"нет чанка с оценкой 3": [], "оценка 3 только у B2B/банка": []}
    off = []  # (чужих чанков в LLM, всего чанков в LLM) на каждый запрос
    start = time.perf_counter()
    for i, item in enumerate(items, 1):
        t = time.perf_counter()
        print(f"[{i}/{len(items)}] {item['id']}:", end="", flush=True)
        results, hr_scored = run_query(item["query"], lambda step: print(f" {step}", end="", flush=True))
        relevant = set(item["relevant_ids"])
        rel_scores.append((item, best_score(hr_scored, relevant), best_score(hr_scored)))
        relevance = item.get("relevance") or {d: 1 for d in relevant}  # старый формат без оценок
        group = doc_groups(item)
        for name, payloads in results.items():
            ranks.setdefault(name, []).append(first_hit(payloads, relevant))
            ndcgs.setdefault(name, []).append(ndcg(payloads, relevance, group))

        answer = answer_ids(item)
        if answer:
            ans_items.append(item)
            for name, payloads in results.items():
                ans_ranks.setdefault(name, []).append(first_answer(payloads, answer))
            if ranks["search→LLM"][-1] and not ans_ranks["search→LLM"][-1]:
                gaps.append(item)
        elif any(j["grade"] == 3 for j in item.get("judgments", [])):
            no_answer_chunk["оценка 3 только у B2B/банка"].append(item["id"])
        else:
            no_answer_chunk["нет чанка с оценкой 3"].append(item["id"])

        llm = results["search→LLM"]
        off.append((sum(map(is_off_segment, llm)), len(llm)))
        hit = "попал" if answer and ans_ranks["search→LLM"][-1] else "промах" if answer else "—"
        print(f"  — {time.perf_counter() - t:.1f} с (ответ в LLM: {hit})")

    na_scores = []
    for i, item in enumerate(no_answer, 1):
        print(f"[без ответа {i}/{len(no_answer)}] {item['id']}: rerank search", end="", flush=True)
        top = best_score(rerank(item["query"], find_candidates(item["query"])))
        n_llm = len(search(item["query"]))
        na_scores.append((item, top, n_llm))
        print(f"  — лучший скор {top:.2f}, в LLM ушло {n_llm} чанков")
    print(f"\nВсего: {time.perf_counter() - start:.0f} с\n")

    n = len(items)
    out(f"Golden set: {GOLDEN_PATH.name}")
    out(f"Запросов с ответом: {n}, без ответа (answerable=false): {len(no_answer)}")
    if skipped:
        out(f"Пропущены — answerable, но relevant_ids пуст (проверить разметку): {', '.join(skipped)}")
    out(f"RETRIEVE_K={RETRIEVE_K}, RERANK_K={RERANK_K}, RERANK_MIN_SCORE={RERANK_MIN_SCORE}")

    out()
    na = len(ans_items)
    out(f"── По абзацу с ответом: чанк с оценкой 3 из b2c-статьи. Запросов: {na} из {n}")
    for reason, ids in no_answer_chunk.items():
        if ids:
            out(f"  не вошли, {reason}: {', '.join(ids)}")
    if na:
        out(f"{'этап':<16}" + "".join(f"{'AHit@' + str(k):>9}" for k in KS) + f"{'MRR':>9}")
        for name, rs in ans_ranks.items():
            hits = [sum(r is not None and r <= k for r in rs) / na for k in KS]
            out(f"{name:<16}" + "".join(f"{h:>9.0%}" for h in hits) + f"{sum(1 / r for r in rs if r) / na:>9.3f}")
        k = sum(r is not None for r in ans_ranks["search→LLM"])
        lo, hi = wilson(k, na)
        out(f"AnswerHit@LLM = {k}/{na} ({k / na:.0%}), 95% ДИ {lo:.0%}–{hi:.0%}")
        out(f"Документ в LLM ушёл, а абзац с ответом — нет: {len(gaps)}")
        for item in gaps:
            out(f"  [{item['id']}] {item['query']}")

    out()
    off_n, total = sum(o for o, _ in off), sum(t for _, t in off)
    out(f"── OffSegment@LLM: B2B и банковских чанков в контексте {off_n}/{total} ({off_n / total:.0%}), "
        f"запросов хотя бы с одним таким: {sum(o > 0 for o, _ in off)}/{n}")

    out()
    out("── По документу (старая метрика, завышена: засчитывает любой абзац релевантной статьи)")
    out(f"{'этап':<16}" + "".join(f"{'Hit@' + str(k):>9}" for k in KS) + f"{'MRR':>9}{'nDCG@' + str(NDCG_K):>9}")
    for name, rs in ranks.items():
        hits = [sum(r is not None and r <= k for r in rs) / n for k in KS]
        mrr = sum(1 / r for r in rs if r) / n
        out(f"{name:<16}" + "".join(f"{h:>9.0%}" for h in hits) + f"{mrr:>9.3f}{sum(ndcgs[name]) / n:>9.3f}")
    out(f"(у search→LLM Hit@k для k > {RERANK_K} совпадает с Hit@{RERANK_K}: в LLM уходит не больше {RERANK_K} чанков)")
    out(f"(1 запрос = {100 / n:.0f} п.п.: разница меньше ~5 п.п. на таком сете — шум)")

    out()
    misses = [(item, r) for item, g, r in zip(items, ranks["search→LLM"], ranks["hybrid+rerank"]) if g is None]
    out(f"Промахи search→LLM (в LLM не ушло ни одного релевантного чанка): {len(misses)}")
    for item, r in misses:
        if r is None:
            reason = f"нет среди {RETRIEVE_K} кандидатов"
        elif r <= RERANK_K:
            reason = f"позиция {r} после реранкера, отрезан порогом"
        else:
            reason = f"позиция {r} после реранкера"
        out(f"  [{item['id']}] {reason}: {item['query']}")

    out()
    out("Калибровка порога реранкера")
    out("  С ответом — скор лучшего релевантного чанка (по возрастанию):")
    for item, rel, top in sorted(rel_scores, key=lambda x: x[1] if x[1] is not None else -1e9)[:10]:
        rel_s = f"{rel:.2f}" if rel is not None else "—"
        out(f"    [{item['id']}] {rel_s:>6} (лучший в выдаче {top:.2f}): {item['query']}")
    out("  Без ответа — скор лучшего чанка в выдаче (по убыванию):")
    for item, top, n_llm in sorted(na_scores, key=lambda x: -x[1]):
        out(f"    [{item['id']}] {top:>6.2f} (сейчас в LLM уходит {n_llm} чанков): {item['query']}")
    if no_answer:
        out()
        out(f"  {'порог':>6}  {'с ответом: пройдёт':>20}  {'без ответа: отсечётся':>22}")
        for t in THRESHOLDS:
            passed = sum(rel is not None and rel >= t for _, rel, _ in rel_scores)
            rejected = sum(top < t for _, top, _ in na_scores)
            out(f"  {t:>6}  {passed:>12}/{n} {passed / n:>5.0%}  {rejected:>14}/{len(no_answer)} {rejected / len(no_answer):>5.0%}")
        out("  (порог хорош, если почти все запросы с ответом проходят, а запросы без ответа отсекаются;")
        out("   если такого порога нет — распределения перекрываются, отказ лучше оставить LLM)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()