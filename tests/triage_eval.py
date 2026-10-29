"""Eval узла классификации triage на размеченном наборе.

Лежит рядом с папкой fixtures. Запуск: run_triage_eval.bat
или из корня support-agent:  python -m <папка>.triage_eval [путь/к/набору.jsonl]

После прогона в results/ появляются два файла:
  triage_<время>_report.txt — сводка: метрики, матрица ошибок, промахи
  triage_<время>.csv        — таблица по кейсам, открывается в Excel
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import llm
from src.agent.state import Context
from src.agent.triage import classify, route_after_triage

FIELDS = ("intent", "wants_operator", "threat")
INTENTS = ("question", "request", "complaint", "smalltalk", "off_topic")
HERE = Path(__file__).parent
CASES_PATH = HERE / "fixtures" / "triage_cases.jsonl"
RESULTS_DIR = HERE / "results"


def to_messages(dialog):
    return [HumanMessage(t) if role == "user" else AIMessage(t) for role, t in dialog]


def route(triage):
    return route_after_triage({"triage": triage})


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def run(path):
    cases = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    # classify берёт из runtime только context.llm
    runtime = SimpleNamespace(context=Context(llm=llm))

    results = []
    for i, case in enumerate(cases, 1):
        t = time.perf_counter()
        got = classify({"messages": to_messages(case["dialog"])}, runtime)["triage"]
        results.append({
            **case,
            "got": got,
            "route_expected": route(case["expected"]),
            "route_got": route(got),
            "seconds": round(time.perf_counter() - t, 2),
        })
        print(f"\r{i}/{len(cases)}", end="", flush=True)
    print()
    return results


def report(results) -> list[str]:
    lines = []
    out = lines.append
    n = len(results)
    failed = [r for r in results if r["got"] is None]
    ok = [r for r in results if r["got"] is not None]

    out(f"══ ИТОГО: {n} кейсов, ошибок LLM: {len(failed)}, "
        f"среднее время: {sum(r['seconds'] for r in results) / n:.1f} с")

    out("\n── Точность по полям (без упавших)")
    for f in FIELDS:
        out(f"  {f:15} {pct(sum(r['got'][f] == r['expected'][f] for r in ok), len(ok))}")
    out(f"  {'маршрут':15} {pct(sum(r['route_got'] == r['route_expected'] for r in results), n)}"
        "  ← главная метрика, упавшие считаются как execute_tools")

    out("\n── Эскалации (маршрут handoff)")
    exp_h = [r for r in results if r["route_expected"] == "handoff"]
    got_h = [r for r in results if r["route_got"] == "handoff"]
    out(f"  recall:          {pct(sum(r['route_got'] == 'handoff' for r in exp_h), len(exp_h))}"
        "  ← пропущенные эскалации, должно быть ~100%")
    out(f"  ложные:          {pct(sum(r['route_expected'] != 'handoff' for r in got_h), len(got_h))}"
        "  ← лишние переводы на оператора")
    for f, values in (("threat", ("violence", "self_harm", "legal")), ("wants_operator", (True,))):
        for v in values:
            sub = [r for r in ok if r["expected"][f] == v]
            out(f"  {f}={v!s:10} {pct(sum(r['got'][f] == v for r in sub), len(sub))}")

    out("\n── Матрица intent (строки — ожидалось, столбцы — получено)")
    m = Counter((r["expected"]["intent"], r["got"]["intent"]) for r in ok)
    out(" " * 12 + "".join(f"{c[:9]:>10}" for c in INTENTS))
    for e in INTENTS:
        out(f"  {e:10}" + "".join(f"{m[(e, g)]:>10}" for g in INTENTS))

    out("\n── Маршрут по тегам (от худшего)")
    by_tag = defaultdict(list)
    for r in results:
        for tag in r["tags"] or ["(без тега)"]:
            by_tag[tag].append(r["route_got"] == r["route_expected"])
    for tag, hits in sorted(by_tag.items(), key=lambda x: sum(x[1]) / len(x[1])):
        out(f"  {tag:15} {pct(sum(hits), len(hits))}")

    out("\n── Промахи (⚠ — ошибка меняет маршрут)")
    for r in results:
        if r["got"] is None:
            out(f"  [{r['id']}] LLM не вернула разметку")
            continue
        diff = [f"{f}: {r['expected'][f]!r}→{r['got'][f]!r}" for f in FIELDS
                if r["got"][f] != r["expected"][f]]
        if diff:
            mark = "  ⚠" if r["route_got"] != r["route_expected"] else ""
            out(f"  [{r['id']}] {', '.join(diff)}{mark}")
            out(f"      сообщение: «{r['dialog'][-1][1]}»")
            out(f"      модель:    {r['got']['reason']}")
    return lines


def save(results, lines):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    report_path = RESULTS_DIR / f"triage_{stamp}_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # utf-8-sig и ";" — чтобы русский Excel открыл двойным кликом без кракозябр
    csv_path = RESULTS_DIR / f"triage_{stamp}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "маршрут верный", "ожидаемый маршрут", "полученный маршрут",
                    *(f"{x} ожид." for x in FIELDS), *(f"{x} получ." for x in FIELDS),
                    "сообщение", "reason модели", "теги", "сек"])
        for r in results:
            got = r["got"] or {}
            w.writerow([r["id"], "да" if r["route_got"] == r["route_expected"] else "НЕТ",
                        r["route_expected"], r["route_got"],
                        *(r["expected"][x] for x in FIELDS), *(got.get(x, "—") for x in FIELDS),
                        r["dialog"][-1][1], got.get("reason", "LLM не вернула разметку"),
                        ", ".join(r["tags"]), r["seconds"]])

    print(f"\nОтчёт:   {report_path}\nТаблица: {csv_path}")


if __name__ == "__main__":
    results = run(sys.argv[1] if len(sys.argv) > 1 else CASES_PATH)
    lines = report(results)
    print("\n".join(lines))
    save(results, lines)