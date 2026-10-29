"""Собирает tests/golden_sets/e2e_set.json из golden set для сквозной оценки.

Запуск один раз: python tests/build_e2e_set.py
Дальше файл правится руками (reference_answer, relevant_ids, verified).
Повторный запуск не перезапишет уже размеченный файл.
"""
import json
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_sets"
SRC = GOLDEN_DIR / "end2end_eval.json"
OUT = GOLDEN_DIR / "e2e_set.json"

# id -> (group, expected_behavior)
SELECTED = {
    **{f"gs-{n:03d}": ("A", "answer") for n in [
        1, 2, 4, 5, 8, 11, 13, 15, 16, 20, 21, 22, 27,
        28, 30, 32, 35, 40, 41, 42, 45, 46, 47, 48, 52, 53,
    ]},
    **{f"gs-{n:03d}": ("B", "refuse") for n in [37, 39, 44, 49, 51]},
    "gs-033": ("C", "correct_premise"),
    "gs-060": ("C", "refuse"),
}

# Подозрительные relevant_ids: убираются, но сохраняются в removed_ids для проверки
SUSPECT = {
    "gs-001": ["domashnii_internet_i_tv/balans-scheta-i-oplata__kogda-spisivayutsya-dengi-za-domashnii-internet-i-tv"],
    "gs-002": ["domashnii_internet_i_tv/balans-scheta-i-oplata__kogda-spisivayutsya-dengi-za-domashnii-internet-i-tv"],
    "gs-005": ["domashnii_internet_i_tv/balans-scheta-i-oplata__kogda-spisivayutsya-dengi-za-domashnii-internet-i-tv"],
    "gs-020": [
        "mts_domashniy_telefon/upravlenie-uslugami__kak-zakazat-detalizatsiyu-ishodyaschih-zvonkov-s-domashnego-telefona",
        "domashnii_internet_i_tv/balans-scheta-i-oplata__kak-proverit-rashodi-za-domashnii-internet-i-tv-msk",
        "domashnii_internet_i_tv/balans-scheta-i-oplata__kak-proverit-rashodi-za-domashnii-internet-i-tv",
    ],
    "gs-028": ["mts_mobilnyy_internet/prodlenie-interneta__kak-podklyuchit-internet"],
    "gs-030": ["kion/ispravlenie-sboev-v-rabote-kion__chto-delat-esli-poyavilis-problemi-so-zvukom-ili-izobrazheniem-na-kion"],
    "gs-041": ["mts-premium/internet-v-podpiske__internet-po-podpiske-dlya-abonentov-mgts-i-drugih-operatorov"],
    "gs-052": [
        "kion_muzika/promokodi-na-kion-muziku__oshibki-pri-aktivatsii-promokodov-na-podpisku-kion-muzika",
        "mts-premium/aktsii-s-mts-premium__mts-premium-pri-oformlenii-debetovoi-karti-mts-dengi",
    ],
}

# Новые вопросы: перед использованием проверить через search(), что ответа в базе нет
NEW = [
    ("e2e-B-01", "B", "refuse", "Как подключить тариф «МТС Космос Безлимит»?"),
    ("e2e-B-02", "B", "refuse", "Сколько стоит минута звонка в роуминге на Мальдивах?"),
    ("e2e-B-03", "B", "refuse", "Есть ли у МТС скидка на тариф для пенсионеров?"),
    ("e2e-B-04", "B", "refuse", "Сколько стоит замена SIM-карты в салоне МТС в Казани?"),
    ("e2e-B-05", "B", "refuse", "Сколько бонусов начисляют за переход в МТС со своим номером?"),
    ("e2e-C-01", "C", "redirect", "Сколько у меня сейчас осталось гигабайт?"),
    ("e2e-C-02", "C", "redirect", "Почему вчера с меня списали 150 рублей?"),
]

if OUT.exists():
    sys.exit(f"{OUT.name} уже существует — удалите его вручную, если нужно пересобрать.")

source = {item["id"]: item for item in json.loads(SRC.read_text(encoding="utf-8"))}

result = []
for item_id, (group, behavior) in SELECTED.items():
    item = source[item_id]
    removed = [i for i in SUSPECT.get(item_id, []) if i in item["relevant_ids"]]
    result.append({
        "id": item_id,
        "group": group,
        "expected_behavior": behavior,
        "query": item["query"],
        "product": item["product"],
        "section": item["section"],
        "relevant_ids": [] if group == "B" else [i for i in item["relevant_ids"] if i not in removed],
        "removed_ids": removed,
        "relevance": item["relevance"],
        "reference_answer": "",
        "verified": False,
        "source": item["source"],
    })

for item_id, group, behavior, query in NEW:
    result.append({
        "id": item_id,
        "group": group,
        "expected_behavior": behavior,
        "query": query,
        "product": "",
        "section": "",
        "relevant_ids": [],
        "removed_ids": [],
        "relevance": {},
        "reference_answer": "",
        "verified": False,
        "source": "manual",
        "check_in_kb": True,
    })

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

counts = {g: sum(r["group"] == g for r in result) for g in "ABC"}
print(f"{OUT.name}: {len(result)} вопросов, {counts}")
