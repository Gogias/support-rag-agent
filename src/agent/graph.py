"""Сборка графа агента.

Запуск из корня support-agent:  python -m src.agent.graph
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.rag import check_guard, execute_tools, generate_answer, route_after_guard
from src.agent.replies import handoff, off_topic, smalltalk
from src.agent.state import AgentState, Context
from src.agent.triage import classify, route_after_triage

builder = StateGraph(AgentState, context_schema=Context)

builder.add_node("classify", classify)
builder.add_node("execute_tools", execute_tools)
builder.add_node("generate_answer", generate_answer)
builder.add_node("check_guard", check_guard)
builder.add_node("smalltalk", smalltalk)
builder.add_node("off_topic", off_topic)
builder.add_node("handoff", handoff)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_after_triage)
builder.add_edge("execute_tools", "generate_answer")
builder.add_edge("generate_answer", "check_guard")
builder.add_conditional_edges("check_guard", route_after_guard)
builder.add_edge("smalltalk", END)
builder.add_edge("off_topic", END)
builder.add_edge("handoff", END)

graph = builder.compile(checkpointer=InMemorySaver())  # чекпоинтер хранит историю диалога


if __name__ == "__main__":
    import logging

    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    from src.agent.llm import llm

    load_dotenv()  # LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST из .env
    logging.basicConfig(level=logging.WARNING)  # только предупреждения и ошибки; ход графа смотреть в Langfuse
    config = {"configurable": {"thread_id": "cli"}, "callbacks": [CallbackHandler()]}
    context = Context(llm=llm)

    while question := input("\nВопрос (пусто — выход)> ").strip():
        result = graph.invoke({"messages": [HumanMessage(question)]}, config, context=context)
        print(f"\n{result['messages'][-1].content}")

    get_client().flush()  # дописать в Langfuse всё, что не успело уйти до выхода