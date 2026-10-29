"""Сквозная оценка, шаг 2: метрики по ручной разметке.

Запуск: run_e2e_report.bat  или  python e2e_report.py  из папки tests.
Сам берёт самый свежий results/eval_results/e2e_*_label.csv
и кладёт рядом <тот же файл>_report.txt.

Правила разметки (только для ответов с got_node = check_guard):
  S / U / C — сколько атомарных утверждений (цифры, сроки, коды, шаги, условия)
              подтверждено контекстом / не найдено в контексте / противоречит ему.
              Сверять с колонкой «контекст», а не с базой и не со своими знаниями.
  correct   — только группа A: c (верно), p (частично), i (неверно) по эталону.
  refused   — y, если бот сказал, что информации нет; иначе n.
"""

import csv
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "eval_results"


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def upper_bound(k, n):
    """Верхняя граница 95% интервала Уилсона для доли k/n."""
    z, p = 1.96, k / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + spread) / (1 + z * z / n)


def read_rows(path):
    # Excel может пересохранить файл в cp1251
    for enc in ("utf-8-sig", "cp1251"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f, delimiter=";"))
        except UnicodeDecodeError:
            continue
    sys.exit(f"Не удалось прочитать {path}")


def num(row, col):
    return int(row[col]) if row[col].strip() else 0


def refused(row):
    return row["refused"].strip().lower() in ("y", "д", "да")


def has_errors(row):
    return num(row, "U") + num(row, "C") > 0


def report(rows):
    lines = []
    out = lines.append

    answered = [r for r in rows if r["got_node"] == "check_guard" and r["group"] in "ABC"]
    labeled = [r for r in answered if any(r[c].strip() for c in ("S", "U", "C"))]
    out(f"══ Ответов через check_guard: {len(answered)}, размечено S/U/C: {len(labeled)}")

    if labeled:
        bad = sum(has_errors(r) for r in labeled)
        s, u, c = (sum(num(r, x) for r in labeled) for x in ("S", "U", "C"))
        out("\n── Главное: ответы без неподтверждённых утверждений")
        out(f"  hallucination-free     {pct(len(labeled) - bad, len(labeled))}")
        out(f"  доля ответов с U/C     {bad}/{len(labeled)}, "
            f"верхняя граница 95%: {upper_bound(bad, len(labeled)):.0%}")
        out(f"  claim precision        {pct(s, s + u + c)}   (S={s}, U={u}, C={c})")

    a = [r for r in rows if r["group"] == "A"]
    out("\n── Группа A: корректность (по вопросу и контексту; по эталону, где он есть)")
    for v, name in (("c", "верно"), ("p", "частично"), ("i", "неверно")):
        out(f"  {name:10} {pct(sum(r['correct'].strip().lower() == v for r in a), len(a))}")
    over = [r for r in a if r["got_node"] == "handoff" or refused(r)]
    out(f"  отказ при наличии ответа  {pct(len(over), len(a))}")

    wrong = [r for r in a if r["correct"].strip().lower() in ("p", "i")]
    out("\n── Откуда ошибки в группе A (частично + неверно); retrieval_hit по статье — завышен")
    out(f"  retrieval не нашёл документ     {sum(r['retrieval_hit'] == 'нет' for r in wrong)}")
    out(f"  документ был, ошибка генерации  {sum(r['retrieval_hit'] == 'да' for r in wrong)}")

    out("\n── Группы B и C (корректно = нужный узел или явный отказ, и ни одного U/C)")
    for g in "BC":
        sub = [r for r in rows if r["group"] == g]
        ok = [r for r in sub if (r["got_node"] == r["expected_node"] or refused(r)) and not has_errors(r)]
        out(f"  группа {g}   {pct(len(ok), len(sub))}")

    out("\n── Классы ошибок: ответов с меткой в notes")
    for tag, name in (("[SEG]", "чужой сегмент (бизнес, банк)"), ("[SVC]", "соседняя услуга"), ("[OTHER]", "остальное")):
        out(f"  {tag:8} {name:30} {sum(tag in r['notes'] for r in labeled)}")

    out("\n── Ответы с U/C — разобрать")
    for r in labeled:
        if has_errors(r):
            out(f"  [{r['id']}] U={num(r, 'U')} C={num(r, 'C')}  {r['notes']}")
    return lines


if __name__ == "__main__":
    files = sorted(RESULTS_DIR.glob("e2e_*_label*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit(f"В {RESULTS_DIR} нет e2e_*_label.csv — сначала запустите run_e2e_eval.bat")
    # берём самый свежий файл, в котором уже есть разметка S/U/C
    labeled = [p for p in files if any(r[c].strip() for r in read_rows(p) for c in ("S", "U", "C"))]
    if not labeled:
        sys.exit(f"Ни в одном файле нет разметки S/U/C. Самый свежий: {files[0].name}")
    path = labeled[0]
    if path != files[0]:
        print(f"Внимание: самый свежий файл {files[0].name} не размечен, беру {path.name}")
    print(f"Разметка: {path.name}\n")
    lines = report(read_rows(path))
    print("\n".join(lines))
    report_path = path.with_name(path.stem + "_report.txt")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт: {report_path}")