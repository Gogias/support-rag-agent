"""
analyze_token_counts.py

Прогоняет все .md файлы из DATA_ROOT через локальный токенизатор bge-m3
и показывает распределение по длине — какие файлы самые тяжёлые, сколько
из них превышают MAX_TOKENS / лимит модели (8192).

Не трогает Qdrant, не эмбеддит — только считает токены и печатает отчёт.
"""

from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# ---- Конфиг — подставь свои пути ------------------------------------------
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_text"
TOKENIZER_PATH = r"D:\ML Enviroment\models\bge-m3-tokenizer"
MODEL_CONTEXT_LIMIT = 8192
MAX_TOKENS = 8000  # текущий порог "чанк = файл целиком" из chunk_and_embed.py
TOP_N = 20  # сколько самых тяжёлых файлов показать в отчёте

_tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def collect_md_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"raw_text не существует: {root}")

    files = sorted(f for f in root.rglob("*.md") if f.name.lower() != "index.md")

    if not files:
        raise FileNotFoundError(f"В {root} не найдено ни одного .md файла (исключая index.md)")

    return files


def main():
    md_files = collect_md_files(DATA_ROOT)
    print(f"Найдено файлов: {len(md_files)}\n")

    results = []
    for md_path in tqdm(md_files, desc="Считаю токены"):
        text = md_path.read_text(encoding="utf-8")
        tokens = count_tokens(text)
        results.append((tokens, md_path))

    results.sort(key=lambda x: x[0], reverse=True)

    total = sum(t for t, _ in results)
    over_context_limit = [r for r in results if r[0] > MODEL_CONTEXT_LIMIT]
    over_max_tokens = [r for r in results if r[0] > MAX_TOKENS]

    print("\n" + "=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"Всего файлов:               {len(results)}")
    print(f"Среднее токенов на файл:    {total / len(results):.0f}")
    print(f"Медиана:                    {results[len(results) // 2][0]}")
    print(f"Минимум:                    {results[-1][0]} ({results[-1][1].name})")
    print(f"Максимум:                   {results[0][0]} ({results[0][1].name})")
    print(f"Файлов > MAX_TOKENS ({MAX_TOKENS}):        {len(over_max_tokens)} — будут разбиты на чанки")
    print(f"Файлов > лимита модели ({MODEL_CONTEXT_LIMIT}):    {len(over_context_limit)} — не влезут в bge-m3 целиком")

    print("\n" + "=" * 70)
    print(f"ТОП-{TOP_N} САМЫХ ТЯЖЁЛЫХ ФАЙЛОВ")
    print("=" * 70)
    for tokens, path in results[:TOP_N]:
        rel = path.relative_to(DATA_ROOT)
        flag = ""
        if tokens > MODEL_CONTEXT_LIMIT:
            flag = "  [!] превышает контекст модели"
        elif tokens > MAX_TOKENS:
            flag = "  [~] будет разбит на чанки"
        print(f"{tokens:>6} токенов  {rel}{flag}")

    if over_context_limit:
        print("\n" + "=" * 70)
        print(f"ВСЕ ФАЙЛЫ, ПРЕВЫШАЮЩИЕ ЛИМИТ МОДЕЛИ ({MODEL_CONTEXT_LIMIT} токенов)")
        print("=" * 70)
        for tokens, path in sorted(over_context_limit, key=lambda x: x[0], reverse=True):
            rel = path.relative_to(DATA_ROOT)
            print(f"{tokens:>6} токенов  {rel}")


if __name__ == "__main__":
    main()