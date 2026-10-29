"""chunk_first, этап 2: GigaChat пишет вопрос клиента к каждому отобранному абзацу.

Запуск: python tests/gen_questions.py
Вход:  tests/golden_sets/chunk_first_candidates.json (этап 1, select_chunks.py)
Выход: tests/golden_sets/chunk_first_draft.json — сохраняется после каждого абзаца,
       повторный запуск продолжит с места остановки.
Креды GigaChat берутся из .env в корне проекта.
"""
import json
import os
import re
from pathlib import Path

from gigachat import GigaChat

HERE = Path(__file__).parent
ENV_PATH = HERE.parent / ".env"
SRC_PATH = HERE / "golden_sets" / "chunk_first_candidates.json"
OUT_PATH = HERE / "golden_sets" / "chunk_first_draft.json"
MODEL = "GigaChat-3-Ultra"

PROMPT = """Ты помогаешь составить проверочный набор для бота поддержки МТС.
Ниже фрагмент статьи из базы знаний. Придумай один вопрос, который мог бы написать в чат поддержки обычный клиент.

Статья: {title}
Раздел: {path}
---
{content}
---

Правила:
1. Ответ на вопрос должен целиком содержаться в этом фрагменте.
2. Не используй слова из названия статьи и официальные названия услуг и разделов. Клиент не знает, как услуга называется в базе, и описывает ситуацию своими словами.
3. Пиши как живой человек в чате: коротко, разговорно, можно без точки в конце. Не начинай с «Подскажите, пожалуйста».
4. Если во фрагменте сказано, что что-то сделать нельзя или невозможно, спроси так, будто клиент хочет это сделать.
5. Если по фрагменту нельзя задать вопрос, на который он отвечает целиком, в строке «Вопрос» напиши НЕТ.

Ответь строго в формате:
Вопрос: <вопрос клиента>
Ответ: <ответ из фрагмента одной-двумя фразами>"""


def load_env():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def ask(giga, prompt):
    if hasattr(giga.chat, "create"):  # новый SDK
        return giga.chat.create(prompt).messages[0].content[0].text
    return giga.chat(prompt).choices[0].message.content  # старый SDK


def field(text, name):
    m = re.search(rf"{name}:\s*(.+)", text)
    return m.group(1).strip() if m else None


def main():
    load_env()
    cands = json.loads(SRC_PATH.read_text(encoding="utf-8"))
    done = {x["cand_id"]: x for x in json.loads(OUT_PATH.read_text(encoding="utf-8"))} if OUT_PATH.exists() else {}

    with GigaChat(model=MODEL, timeout=120, max_retries=5, retry_backoff_factor=2) as giga:
        for i, c in enumerate(cands, 1):
            if c["cand_id"] in done:
                continue
            text = ask(giga, PROMPT.format(title=c.get("title") or "", content=c["content"],
                                           path=" > ".join(c.get("heading_path") or [])))
            question = field(text, "Вопрос")
            done[c["cand_id"]] = {**c, "question": None if question in (None, "НЕТ") else question,
                                  "answer": field(text, "Ответ"), "raw": text}
            print(f"[{i}/{len(cands)}] {c['cand_id']}: {question}")
            OUT_PATH.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    empty = [x["cand_id"] for x in done.values() if not x["question"]]
    print(f"\nГотово: {len(done) - len(empty)} вопросов, без вопроса: {len(empty)}\nСохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
