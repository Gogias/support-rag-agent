"""Единственное место, где создаётся LLM. Узлы импортируют её отсюда."""

from langchain_openai import ChatOpenAI

from src.config import settings

llm = ChatOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
    temperature=0.2,
    max_tokens=1024,
    timeout=300,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)