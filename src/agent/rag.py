import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime

from src.agent.retrieval import search
from src.agent.state import AgentState, Context
from src.prompt_loader import load_prompt

log = logging.getLogger(__name__)

PROMPT = load_prompt("rag")
CITATION = re.compile(r"\[(\d+)\]")


def last_question(state: AgentState) -> str:
    return next(m.content for m in reversed(state["messages"]) if m.type == "human")


def question_for_search(state: AgentState) -> str:
    """Последняя реплика; для продолжения диалога («а на 50?») — её самостоятельная версия из триажа.
    На самостоятельных вопросах переформулировка не используется: там она не помогала, а иногда вредила."""
    has_history = sum(isinstance(m, HumanMessage) for m in state["messages"]) > 1
    rewritten = (state["triage"] or {}).get("search_query") if has_history else None
    return rewritten or last_question(state)


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for n, c in enumerate(chunks, 1):
        parts = [f"[{n}] " + " > ".join([c["title"], *c["heading_path"]])]
        if c["before"]:
            parts.append(f"(предыдущий фрагмент) {c['before']}")
        parts.append(c["content"])
        if c["after"]:
            parts.append(f"(следующий фрагмент) {c['after']}")
        blocks.append("\n".join(parts))
    return "\n\n---\n\n".join(blocks)


def cited(draft: str) -> set[int]:
    return {int(n) for n in CITATION.findall(draft)}


def validate_response(draft: str, chunks: list[dict]) -> bool:
    refs = cited(draft)
    return bool(refs) and refs <= set(range(1, len(chunks) + 1))


def with_sources(draft: str, chunks: list[dict]) -> str:
    lines = [f"[{n}] {', '.join(chunks[n - 1]['sources'])}" for n in sorted(cited(draft))]
    return draft + "\n\nИсточники:\n" + "\n".join(lines)


def execute_tools(state: AgentState) -> dict:
    try:
        return {"tool_results": search(question_for_search(state))}
    except Exception:
        log.exception("search failed")
        return {"tool_results": []}


def generate_answer(state: AgentState, runtime: Runtime[Context]) -> dict:
    chunks = state["tool_results"]
    if not chunks:
        return {"draft": ""}
    user = PROMPT["user"].format(context=build_context(chunks), question=question_for_search(state))
    try:
        reply = runtime.context.llm.invoke([SystemMessage(PROMPT["system"]), HumanMessage(user)])
    except Exception:
        log.exception("generation failed")
        return {"draft": ""}
    return {"draft": reply.content}


def check_guard(state: AgentState) -> dict:
    draft, chunks = state["draft"], state["tool_results"]
    if validate_response(draft, chunks):
        return {"messages": [AIMessage(with_sources(draft, chunks))], "needs_escalation": False}
    return {"needs_escalation": True}


def route_after_guard(state: AgentState) -> Literal["handoff", "__end__"]:
    return "handoff" if state["needs_escalation"] else END