"""
Configuração global do Pytest e Fixtures reutilizáveis.

Este arquivo centraliza mocks, utilitários de simulação de respostas da OpenAI
e fixtures de ambiente para viabilizar testes determinísticos sem depender de
chamadas de rede ou custos de API na esteira de CI.
"""

import json
from typing import Any, Optional
import pytest


class MockFunctionCall:
    """Simula o objeto function dentro de uma ToolCall da OpenAI."""

    def __init__(self, name: str, arguments: dict[str, Any] | str):
        self.name = name
        self.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)


class MockToolCall:
    """Simula uma chamada de ferramenta retornada pela OpenAI."""

    def __init__(self, tool_id: str, name: str, arguments: dict[str, Any] | str):
        self.id = tool_id
        self.type = "function"
        self.function = MockFunctionCall(name=name, arguments=arguments)


class MockMessage:
    """Simula a mensagem retornada por choices[0].message da OpenAI."""

    def __init__(self, content: Optional[str] = None, tool_calls: Optional[list[MockToolCall]] = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        """Simula a serialização da mensagem no formato esperado por ChatCompletionMessageParam."""
        dumped: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            return {k: v for k, v in dumped.items() if v is not None}
        return dumped


class MockChoice:
    """Simula um choice da resposta de ChatCompletion."""

    def __init__(self, message: MockMessage):
        self.message = message


class MockChatCompletionResponse:
    """Simula a resposta completa retornada por client.chat.completions.create()."""

    def __init__(self, content: Optional[str] = None, tool_calls: Optional[list[MockToolCall]] = None):
        self.choices = [MockChoice(MockMessage(content=content, tool_calls=tool_calls))]


@pytest.fixture
def mock_openai_response():
    """Fábrica de respostas simuladas da OpenAI para testes."""

    def _criar(content: Optional[str] = None, tool_calls: Optional[list[MockToolCall]] = None):
        return MockChatCompletionResponse(content=content, tool_calls=tool_calls)

    return _criar


@pytest.fixture
def fake_env(monkeypatch):
    """Garante variáveis de ambiente seguras e fake para evitar chamadas acidentais a APIs reais."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-local-testing-123456789")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("AGENTE_TEMPERATURA", "0.7")
    monkeypatch.setenv("AGENTE_MAX_ITERACOES", "5")
