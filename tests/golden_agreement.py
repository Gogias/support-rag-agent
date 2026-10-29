"""Согласие LLM-судьи с человеком на выборке из review_golden.py.

Запуск: python tests/golden_agreement.py
Отчёт: tests/results/eval_results/agreement_<дата-время>.txt
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

PATH = Path(__file__).parent / "golden_sets" / "judge_review_sample.json"
RESULTS_DIR = Path(__file__).parent / "results" / "eval_results"
REL_MIN = 2  # с какой оценки чанк релевантен (как в relabel_golden.py)
LABELS = range(4)


def kappa(pairs, labels, weight):
    """Cohen's kappa; weight(i, j) — степень согласия меток i и j (1 — полное)."""
    n = len(pairs)
    obs, a, b = Counter(pairs), Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    po = sum(weight(i, j) * obs[i, j] for i in labels for j in labels) / n
    pe = sum(weight(i, j) * a[i] * b[j] for i in labels for j in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def main():
    lines = []
    out = lines.append
    rated = [s for s in json.loads(PATH.read_text(encoding="utf-8")) if "human_grade" in s]
    pairs = [(s["judge_grade"], s["human_grade"]) for s in rated]
    binary = [(int(j >= REL_MIN), int(h >= REL_MIN)) for j, h in pairs]

    judge_rel = [h for j, h in binary if j]
    judge_not = [h for j, h in binary if not j]
    out(f"Выборка: {len(rated)} чанков ({PATH.name}), граница релевантности: оценка >= {REL_MIN}")
    out(f"Точность «релевантно» у судьи:  {pct(sum(judge_rel), len(judge_rel))}"
        "  ← человек согласен, что чанк релевантен")
    out(f"Пропуски судьи:                 {pct(sum(judge_not), len(judge_not))}"
        "  ← судья поставил 0–1, человек счёл релевантным")
    out(f"Совпадение «релевантно / нет»:  {pct(sum(j == h for j, h in binary), len(binary))}")
    out(f"Cohen's kappa (релевантно/нет): {kappa(binary, (0, 1), lambda i, j: float(i == j)):.2f}")
    out(f"Weighted kappa 0–3:             {kappa(pairs, LABELS, lambda i, j: 1 - (i - j) ** 2 / 9):.2f}")
    out("(kappa: 0.4–0.6 умеренное, 0.6–0.8 существенное, > 0.8 почти полное согласие)")
    out("(выборка стратифицирована по оценке судьи, поэтому kappa — ориентир, главное — две строки выше)")

    out("\nСогласие по оценке судьи:")
    for g in LABELS:
        sub = [h for j, h in pairs if j == g]
        if sub:
            out(f"  судья {g}: человек ставит то же {pct(sum(h == g for h in sub), len(sub))}, "
                f"релевантно (>= {REL_MIN}) {pct(sum(h >= REL_MIN for h in sub), len(sub))}")

    out("\nМатрица: строки — судья, столбцы — человек")
    m = Counter(pairs)
    out("      " + "".join(f"{h:>6}" for h in LABELS))
    for g in LABELS:
        out(f"{g:>6}" + "".join(f"{m[g, h]:>6}" for h in LABELS))

    out(f"\nРасхождения через границу {REL_MIN} (судья → человек):")
    for s in rated:
        if (s["judge_grade"] >= REL_MIN) != (s["human_grade"] >= REL_MIN):
            out(f"  [{s['id']}] {s['judge_grade']} → {s['human_grade']}  {s['chunk_id']}")
            out(f"      {s['query']}")

    print("\n".join(lines))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"agreement_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()