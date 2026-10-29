from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Triage(BaseModel):
    reason: str = Field(description="Кратко: почему такая классификация")
    intent: Literal["question", "request", "complaint", "smalltalk", "off_topic"]
    wants_operator: bool = Field(description="Пользователь явно просит живого человека")
    threat: Literal["none", "violence", "self_harm", "legal"]
    search_query: str = Field(description="Только для продолжения диалога: самостоятельный вопрос по истории. Иначе пустая строка")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    triage: dict | None  # Triage.model_dump(): в чекпоинтах храним простые типы
    tool_results: list[dict]
    draft: str  # ответ до проверки guard; в messages попадает только после неё
    needs_escalation: bool


@dataclass
class Context:
    """Зависимости графа: передаются при запуске, в чекпоинты не попадают."""
    llm: BaseChatModel