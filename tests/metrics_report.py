"""Эксплуатационные метрики и macro-F1 по уже сохранённым прогонам. Модели не вызываются.

Запуск: python tests/metrics_report.py
Сам берёт самые свежие файлы:
  results/triage_*.csv              — из triage_eval.py: macro-F1 и P/R/F1 по intent, задержка триажа
  results/eval_results/e2e_*.jsonl  — из e2e_eval.py: задержка p50/p95 (всего и по узлам),
                                       токены по узлам, answer rate, context precision
Время и токены по узлам есть только в прогонах e2e_eval.py после правки от 26.10.
Отчёт: results/eval_results/metrics_<дата-время>.txt
"""
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.ingestion.segment import segment  # noqa: E402

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
EVAL_DIR = RESULTS_DIR / "eval_results"
INTENTS = ("question", "request", "complaint", "smalltalk", "off_topic")
CONTEXT_LIMIT = 8192
NEAR_LIMIT = 7000   # input генерации выше этого — контекст близок к лимиту
REL_MIN = 2         # документ в контексте релевантен с оценки 2 (как в relabel_golden.py)


def latest(folder, pattern):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def q(values, p):
    """Перцентиль p (0–100) методом ближайшего ранга."""
    v = sorted(values)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


def p50_p95(values, fmt="{:.1f}"):
    return f"{fmt.format(q(values, 50))} / {fmt.format(q(values, 95))}" if values else "—"


def triage_section(path, out):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    ok = [r for r in rows if r["intent получ."] in INTENTS]
    out(f"══ Триаж: {path.name}, кейсов {len(rows)}, без разметки LLM {len(rows) - len(ok)}")

    out("\n── intent: precision / recall / F1")
    pairs = Counter((r["intent ожид."], r["intent получ."]) for r in ok)
    f1s = []
    for c in INTENTS:
        tp = pairs[c, c]
        got = sum(n for (_, g), n in pairs.items() if g == c)
        exp = sum(n for (e, _), n in pairs.items() if e == c)
        p, r = (tp / got if got else 0), (tp / exp if exp else 0)
        f1 = 2 * p * r / (p + r) if p + r else 0
        f1s.append(f1)
        out(f"  {c:10}  P {p:.2f}  R {r:.2f}  F1 {f1:.2f}  (ожидалось {exp})")
    acc = sum(pairs[c, c] for c in INTENTS) / len(ok)
    out(f"  accuracy {acc:.0%}, macro-F1 {sum(f1s) / len(f1s):.2f}")

    secs = [float(r["сек"].replace(",", ".")) for r in rows if r["сек"]]
    out(f"\n  задержка триажа p50 / p95: {p50_p95(secs)} с")


def e2e_section(path, out):
    rs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    out(f"\n══ Сквозной прогон: {path.name}, кейсов {len(rs)}")

    out("\n── Задержка, с (p50 / p95)")
    out(f"  весь граф        {p50_p95([r['seconds'] for r in rs])}")
    answered = [r for r in rs if "generate_answer" in r["path"]]
    out(f"  с генерацией     {p50_p95([r['seconds'] for r in answered])}  ← кейсы, дошедшие до ответа")
    nodes = [r["node_seconds"] for r in rs if r.get("node_seconds")]
    if nodes:
        for node in ("classify", "execute_tools", "generate_answer", "check_guard"):
            vals = [n[node] for n in nodes if node in n]
            out(f"  {node:16} {p50_p95(vals, '{:.2f}')}")
    else:
        out("  по узлам: нет node_seconds — прогон сделан до правки e2e_eval.py")

    out("\n── Токены (p50 / p95)")
    tokens = [r["tokens"] for r in rs if r.get("tokens")]
    if tokens:
        for node in ("classify", "generate_answer"):
            ins = [t[node]["input"] for t in tokens if t.get(node)]
            outs = [t[node]["output"] for t in tokens if t.get(node)]
            out(f"  {node:16} input {p50_p95(ins, '{:.0f}')}, output {p50_p95(outs, '{:.0f}')}")
        total = [sum(u["input"] + u["output"] for u in t.values()) for t in tokens]
        out(f"  {'на обращение':16} {p50_p95(total, '{:.0f}')}")
        gen_in = [t["generate_answer"]["input"] for t in tokens if t.get("generate_answer")]
        near = sum(x > NEAR_LIMIT for x in gen_in)
        out(f"  input генерации > {NEAR_LIMIT} (лимит {CONTEXT_LIMIT}): {pct(near, len(gen_in))}, "
            f"максимум {max(gen_in, default=0)}")
    else:
        out("  нет tokens — прогон сделан до правки e2e_eval.py или бэкенд не вернул usage")

    a = [r for r in rs if r["group"] == "A"]
    out("\n── Группа A")
    out(f"  answer rate      {pct(sum(r['got_node'] == 'check_guard' for r in a), len(a))}"
        "  ← ответ дошёл до клиента, а не ушёл в handoff / off_topic / smalltalk")
    stopped = sum("generate_answer" in r["path"] and r["got_node"] == "handoff" for r in a)
    out(f"  guard остановил  {pct(stopped, len(a))}  ← черновики есть в поле draft, их можно разметить S/U/C")

    # Context precision по документам: доля документов в контексте с оценкой >= REL_MIN из b2c.
    # Неразмеченные документы считаются нерелевантными (pool bias), поэтому это оценка снизу.
    hit, total, per_case = 0, 0, []
    for r in a:
        docs = set(r.get("sources") or [])
        if not docs or not r.get("relevance"):
            continue
        rel = {d for d in docs if r["relevance"].get(d, 0) >= REL_MIN and segment(d) == "b2c"}
        hit, total = hit + len(rel), total + len(docs)
        per_case.append(len(rel) / len(docs))
    out(f"  context precision {pct(hit, total)} документов в контексте, "
        f"среднее по запросам {sum(per_case) / len(per_case):.0%}" if per_case else "  context precision —")
    out("  (оценка снизу: документы без оценки судьи считаются нерелевантными)")


def main():
    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    triage = latest(RESULTS_DIR, "triage_*.csv")
    e2e = latest(EVAL_DIR, "e2e_*.jsonl")
    if triage:
        triage_section(triage, out)
    else:
        out(f"Нет {RESULTS_DIR}/triage_*.csv — сначала python -m tests.triage_eval")
    if e2e:
        e2e_section(e2e, out)
    else:
        out(f"\nНет {EVAL_DIR}/e2e_*.jsonl — сначала e2e_eval.py")

    path = EVAL_DIR / f"metrics_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
