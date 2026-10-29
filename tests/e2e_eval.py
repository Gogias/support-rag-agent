"""Сквозная оценка графа, шаг 1: прогон.

Запуск: run_e2e_eval.bat  или  python e2e_eval.py  из папки tests.
Берёт golden_sets/e2e_set.json, гонит каждый вопрос через весь граф
и кладёт в results/eval_results/:
  e2e_<время>.jsonl       — всё по кейсу: путь по узлам, triage, черновик, ответ, контекст
  e2e_<время>_label.csv   — таблица для ручной разметки (колонки S, U, C, correct, refused, notes)
  e2e_<время>_report.txt  — автоматические метрики: маршрут, retrieval hit, guard

Шаг 2 — e2e_report.py: считает метрики по последнему размеченному _label.csv.
"""

import csv
import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from src.agent.graph import graph  # noqa: E402
from src.agent.llm import llm  # noqa: E402
from src.agent.rag import build_context  # noqa: E402
from src.agent.state import Context  # noqa: E402

HERE = Path(__file__).parent
SET_PATH = HERE / "golden_sets" / "e2e_set.json"
RESULTS_DIR = HERE / "results" / "eval_results"
NODES = ("check_guard", "handoff", "off_topic", "smalltalk")
EXCEL_CELL_LIMIT = 32000
LABEL_COLS = ("S", "U", "C", "correct", "refused", "notes")


class TokenCounter(BaseCallbackHandler):
    """Токены каждого вызова LLM по узлу графа: {"classify": {"input": .., "output": ..}, ...}."""

    def __init__(self):
        self.nodes, self.by_node = {}, {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self.nodes[run_id] = (metadata or {}).get("langgraph_node", "?")

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = getattr(response.generations[0][0], "message", None)
        usage = getattr(usage, "usage_metadata", None)
        if usage:
            node = self.by_node.setdefault(self.nodes.get(run_id, "?"), {"input": 0, "output": 0})
            node["input"] += usage["input_tokens"]
            node["output"] += usage["output_tokens"]


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def run_case(item, context):
    tokens = TokenCounter()
    config = {"configurable": {"thread_id": f"e2e-{item['id']}-{uuid.uuid4().hex[:6]}"},
              "callbacks": [tokens]}
    path, node_seconds = [], {}
    t = last = time.perf_counter()
    try:
        for update in graph.stream({"messages": [HumanMessage(item["query"])]}, config,
                                   context=context, stream_mode="updates"):
            path.extend(update.keys())
            now = time.perf_counter()
            for node in update:  # update приходит, когда узел закончил работу
                node_seconds[node] = round(now - last, 2)
            last = now
        error = None
    except Exception as e:  # кейс падает, прогон продолжается
        error = repr(e)
    seconds = round(time.perf_counter() - t, 1)

    values = graph.get_state(config).values
    chunks = values.get("tool_results") or []
    sources = [s for c in chunks for s in c["sources"]]
    relevant = set(item["relevant_ids"])
    messages = values.get("messages") or []
    return {
        **item,
        "path": path,
        "got_node": path[-1] if path else "error",
        "triage": values.get("triage"),
        "draft": values.get("draft", ""),
        "answer": messages[-1].content if len(messages) > 1 else "",
        "context": build_context(chunks) if chunks else "",
        "sources": sources,
        "retrieval_hit": bool(relevant & set(sources)) if relevant else None,
        "seconds": seconds,
        "node_seconds": node_seconds,
        "tokens": tokens.by_node,
        "error": error,
    }


def report(results):
    lines = []
    out = lines.append
    n = len(results)
    out(f"══ {n} кейсов, ошибок: {sum(bool(r['error']) for r in results)}, "
        f"среднее время: {sum(r['seconds'] for r in results) / n:.1f} с")

    out("\n── Финальный узел совпал с ожидаемым")
    out(f"  всего        {pct(sum(r['got_node'] == r['expected_node'] for r in results), n)}")
    for g in sorted({r["group"] for r in results}):
        sub = [r for r in results if r["group"] == g]
        out(f"  группа {g}     {pct(sum(r['got_node'] == r['expected_node'] for r in sub), len(sub))}")

    out("\n── Матрица узлов (строки — ожидалось, столбцы — получено)")
    m = Counter((r["expected_node"], r["got_node"]) for r in results)
    cols = (*NODES, "error")
    out(" " * 14 + "".join(f"{c[:11]:>12}" for c in cols))
    for e in NODES:
        out(f"  {e:12}" + "".join(f"{m[(e, g)]:>12}" for g in cols))

    a = [r for r in results if r["group"] == "A"]
    searched = [r for r in a if "execute_tools" in r["path"]]
    out("\n── Группа A")
    out(f"  дошли до поиска            {pct(len(searched), len(a))}")
    out(f"  retrieval hit              {pct(sum(bool(r['retrieval_hit']) for r in searched), len(searched))}"
        "  ← среди дошедших до поиска")
    out(f"  ответ прошёл guard         {pct(sum(r['got_node'] == 'check_guard' for r in a), len(a))}")
    out(f"  guard отправил в handoff   "
        f"{pct(sum('generate_answer' in r['path'] and r['got_node'] == 'handoff' for r in a), len(a))}"
        "  ← черновик без цитат")

    out("\n── Не совпал узел")
    for r in results:
        if r["got_node"] != r["expected_node"]:
            intent = (r["triage"] or {}).get("intent", "—")
            out(f"  [{r['id']}] {r['expected_node']} → {r['got_node']}  (intent={intent})  «{r['query']}»")
    return lines


def save(results, lines):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"e2e_{datetime.now():%Y%m%d_%H%M%S}"

    with open(f"{base}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # utf-8-sig и ";" — чтобы русский Excel открыл двойным кликом
    with open(f"{base}_label.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "group", "expected_behavior", "expected_node", "got_node", "retrieval_hit",
                    "вопрос", "ответ", "эталон", "контекст", *LABEL_COLS])
        for r in results:
            w.writerow([r["id"], r["group"], r["expected_behavior"], r["expected_node"], r["got_node"],
                        {True: "да", False: "нет", None: ""}[r["retrieval_hit"]],
                        r["query"], r["answer"], r["reference_answer"],
                        r["context"][:EXCEL_CELL_LIMIT], *[""] * len(LABEL_COLS)])

    Path(f"{base}_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nДанные:   {base}.jsonl\nРазметка: {base}_label.csv\nОтчёт:    {base}_report.txt")


if __name__ == "__main__":
    items = json.loads(SET_PATH.read_text(encoding="utf-8"))
    # temperature=0, чтобы повторный прогон был сравним с предыдущим
    context = Context(llm=llm.model_copy(update={"temperature": 0}))
    results = []
    for i, item in enumerate(items, 1):
        results.append(run_case(item, context))
        print(f"\r{i}/{len(items)} {item['id']}", end="", flush=True)
    print()
    lines = report(results)
    print("\n".join(lines))
    save(results, lines)