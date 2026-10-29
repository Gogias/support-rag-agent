import logging
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from src.agent.state import AgentState, Context, Triage
from src.prompt_loader import load_prompt

log = logging.getLogger(__name__)

TRIAGE = load_prompt("triage")
HISTORY_LIMIT = 6
BOT_MSG_LIMIT = 300  # классификатору не нужен полный ответ бота, экономим контекст
ROLES = {"human": "Пользователь", "ai": "Бот"}


def format_history(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages[-HISTORY_LIMIT:]:
        if m.type not in ROLES:
            continue
        # не даём пользователю закрыть тег <dialog> и выйти из блока данных
        text = m.content.replace("<dialog>", "").replace("</dialog>", "")
        if m.type == "ai" and len(text) > BOT_MSG_LIMIT:
            text = text[:BOT_MSG_LIMIT] + "…"
        lines.append(f"{ROLES[m.type]}: {text}")
    return "\n".join(lines)


def classify(state: AgentState, runtime: Runtime[Context]) -> dict:
    try:
        triage = runtime.context.llm.with_structured_output(Triage, method="json_schema").invoke([
            SystemMessage(TRIAGE["system"]),
            HumanMessage(TRIAGE["user"].format(history=format_history(state["messages"]))),
        ])
    except Exception:
        log.exception("triage failed")
        triage = None
    return {"triage": triage.model_dump() if triage else None}


def route_after_triage(state: AgentState) -> Literal["handoff", "off_topic", "smalltalk", "execute_tools"]:
    t = state["triage"]
    if t is None:
        return "execute_tools"
    if t["threat"] in ("violence", "self_harm") or t["wants_operator"]:
        return "handoff"
    if t["intent"] == "off_topic":
        return "off_topic"
    if t["intent"] == "smalltalk":
        return "smalltalk"
    return "execute_tools"