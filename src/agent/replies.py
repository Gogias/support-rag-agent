from langchain_core.messages import AIMessage

from src.agent.state import AgentState


def smalltalk(state: AgentState) -> dict:
    return {"messages": [AIMessage("Я помощник поддержки МТС. Чем могу помочь?")]}


def off_topic(state: AgentState) -> dict:
    return {"messages": [AIMessage("Я отвечаю только на вопросы об услугах МТС.")]}


def handoff(state: AgentState) -> dict:
    t = state.get("triage")
    if t and t["threat"] == "self_harm":
        text = "Мне жаль, что вам сейчас тяжело. Если вы в опасности, позвоните 112. Подключаю живого человека."
    else:
        text = "Передаю ваше обращение оператору."
    return {"messages": [AIMessage(text)], "needs_escalation": True}
