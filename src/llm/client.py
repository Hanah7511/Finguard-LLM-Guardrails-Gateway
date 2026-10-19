"""LLM client: LangChain + OpenAI. Any object with ``complete`` can be swapped in."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class LLMCompletion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion: ...


class LLMUnavailableError(RuntimeError):
    """No LLM configured (e.g. OPENAI_API_KEY unset)."""


class OpenAIChatClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, timeout: float = 30.0) -> None:
        from langchain_openai import ChatOpenAI

        self._model_name = model
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            timeout=timeout,
            max_retries=1,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        from langchain_core.messages import HumanMessage, SystemMessage

        message = self._llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        usage = getattr(message, "usage_metadata", None) or {}
        model = (getattr(message, "response_metadata", None) or {}).get("model_name", self._model_name)
        return LLMCompletion(
            text=message.content if isinstance(message.content, str) else str(message.content),
            model=model,
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )


def build_llm_client(api_key: Optional[str], model: str, temperature: float, timeout: float) -> Optional[LLMClient]:
    if not api_key:
        return None
    return OpenAIChatClient(api_key, model, temperature, timeout)
