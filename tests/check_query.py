"""Прогон одного запроса через весь граф: triage → retrieval → ответ LLM.
Печатает, что вернул каждый узел, поэтому виден и контекст, и ответ.

Запуск из корня проекта:
    python tests/check_query.py "как подключить роуминг"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from src.agent.graph import graph  # noqa: E402  ← поправь путь/имя, если граф называется иначе

PREVIEW_CHARS = 500  # обрезка чанков контекста; ответ LLM печатается целиком


def fmt(value):
    if isinstance(value, list):
        return "\n".join(f"  [{i}] {fmt(v)}" for i, v in enumerate(value, 1))
    if hasattr(value, "page_content"):  # Document — это контекст
        meta = "  ".join(f"{k}={v}" for k, v in value.metadata.items())
        text = value.page_content.replace("\n", " ")
        return f"{meta}\n      {text[:PREVIEW_CHARS]}{'…' if len(text) > PREVIEW_CHARS else ''}"
    if hasattr(value, "content"):  # сообщение — ответ LLM
        return f"{type(value).__name__}: {value.content}"
    return str(value)


def main():
    query = " ".join(sys.argv[1:]) or input("Запрос: ")
    inputs = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "check_query"}}

    for step in graph.stream(inputs, config, stream_mode="updates"):
        for node, update in step.items():
            print(f"\n{'=' * 20} {node} {'=' * 20}")
            for key, value in (update or {}).items():
                print(f"{key}:\n{fmt(value)}" if isinstance(value, list) else f"{key}: {fmt(value)}")


if __name__ == "__main__":
    main()