"""
==============================================================================
TÓPICO 3: TESTES DE INTEGRAÇÃO & MOCKS
==============================================================================

Objetivo:
Validar o ciclo completo do agente (`responder`) que coordena a conversa com a
OpenAI e a execução local de ferramentas. Como chamadas reais ao modelo têm
latência, custo e variabilidade estocástica, os testes de integração usam mocks
precisos para simular o protocolo da OpenAI e validar determinísticamente o
comportamento do loop de raciocínio.

O que é testado aqui:
1. Resposta Direta em Texto:
   - Quando o modelo devolve texto sem `tool_calls`, o loop encerra no primeiro
     passo e atualiza o histórico com o papel de 'assistant'.
2. Orquestração de Tool Calling em Múltiplos Passos:
   - O modelo devolve uma chamada de ferramenta (`somar`).
   - O agente intercepta, executa o código Python local e anexa a resposta
     com papel 'tool'.
   - O modelo recebe o resultado e formula a resposta final ao usuário.
3. Múltiplas Chamadas de Ferramenta em uma Única Rodada:
   - O modelo solicita duas chamadas simultâneas (ex: duas somas distintas).
   - O agente processa ambas e adiciona as duas mensagens 'tool' correspondentes.
4. Resiliência a Argumentos Malformados:
   - Simulação de JSON corrompido retornado pelo modelo nos argumentos da função.
5. Proteção contra Loops Infinitos (`MAX_ITERACOES`):
   - Se o modelo continuar requisitando ferramentas além do limite configurado,
     o agente interrompe a execução com uma mensagem explicativa.
6. Smoke Test (Opcional, com API Real):
   - Executado apenas quando `OPENAI_API_KEY` válida estiver configurada no ambiente.
"""

import json
import os
from unittest.mock import MagicMock
import pytest

import agent
from tests.conftest import MockToolCall


@pytest.mark.integration
class TestLoopDoAgenteComMocks:
    """Testes de integração do loop de execução `responder` com simulação da API OpenAI."""

    def test_resposta_direta_em_texto(self, mock_openai_response, monkeypatch):
        """Valida caso simples onde o modelo não necessita de ferramentas."""
        cliente_mock = MagicMock()
        cliente_mock.chat.completions.create.return_value = mock_openai_response(
            content="Olá! Como posso ajudar você hoje?"
        )
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        mensagens = [{"role": "user", "content": "Olá"}]
        resposta = agent.responder(mensagens)

        assert resposta == "Olá! Como posso ajudar você hoje?"
        assert len(mensagens) == 2
        assert mensagens[-1]["role"] == "assistant"
        assert mensagens[-1]["content"] == "Olá! Como posso ajudar você hoje?"
        cliente_mock.chat.completions.create.assert_called_once()

    def test_fluxo_completo_com_chamada_de_ferramenta(self, mock_openai_response, monkeypatch):
        """Valida o fluxo em 2 etapas: (1) modelo pede ferramenta -> (2) modelo recebe retorno e responde."""
        cliente_mock = MagicMock()

        # Turno 1: modelo pede para somar 15 + 25
        chamada = MockToolCall(tool_id="call_abc123", name="somar", arguments={"a": 15, "b": 25})
        resposta_passo_1 = mock_openai_response(content=None, tool_calls=[chamada])

        # Turno 2: modelo recebe resultado e responde em texto
        resposta_passo_2 = mock_openai_response(content="A soma de 15 e 25 é 40.")

        cliente_mock.chat.completions.create.side_effect = [resposta_passo_1, resposta_passo_2]
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        mensagens = [{"role": "user", "content": "Quanto é 15 + 25?"}]
        resposta_final = agent.responder(mensagens, verboso=False)

        assert resposta_final == "A soma de 15 e 25 é 40."
        assert cliente_mock.chat.completions.create.call_count == 2

        # Valida histórico completo: user -> assistant (tool_calls) -> tool -> assistant (texto)
        assert len(mensagens) == 4
        assert mensagens[0]["role"] == "user"
        assert mensagens[1]["role"] == "assistant"
        assert "tool_calls" in mensagens[1]
        assert mensagens[2]["role"] == "tool"
        assert mensagens[2]["tool_call_id"] == "call_abc123"
        assert json.loads(mensagens[2]["content"]) == {"resultado": 40}
        assert mensagens[3]["role"] == "assistant"
        assert mensagens[3]["content"] == "A soma de 15 e 25 é 40."

    def test_multiplas_chamadas_de_ferramenta_em_um_turno(self, mock_openai_response, monkeypatch):
        """Valida que o agente lida com múltiplas chamadas simultâneas de ferramentas."""
        cliente_mock = MagicMock()

        chamada_1 = MockToolCall(tool_id="call_1", name="somar", arguments={"a": 1, "b": 2})
        chamada_2 = MockToolCall(tool_id="call_2", name="somar", arguments={"a": 3, "b": 4})
        resposta_passo_1 = mock_openai_response(content=None, tool_calls=[chamada_1, chamada_2])
        resposta_passo_2 = mock_openai_response(content="Os resultados são 3 e 7.")

        cliente_mock.chat.completions.create.side_effect = [resposta_passo_1, resposta_passo_2]
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        mensagens = [{"role": "user", "content": "Some 1+2 e depois some 3+4"}]
        resposta = agent.responder(mensagens)

        assert resposta == "Os resultados são 3 e 7."
        # Histórico deve conter: user + assistant + tool_1 + tool_2 + assistant final = 5 mensagens
        assert len(mensagens) == 5
        mensagens_tool = [m for m in mensagens if m.get("role") == "tool"]
        assert len(mensagens_tool) == 2
        assert json.loads(mensagens_tool[0]["content"]) == {"resultado": 3}
        assert json.loads(mensagens_tool[1]["content"]) == {"resultado": 7}

    def test_tratamento_de_argumentos_json_corrompidos(self, mock_openai_response, monkeypatch):
        """Valida resiliência quando a OpenAI envia uma string de argumentos com JSON inválido."""
        cliente_mock = MagicMock()

        # Argumento com JSON quebrado
        chamada = MockToolCall(tool_id="call_quebrada", name="somar", arguments="JSON_INVALIDO{{{")
        resposta_passo_1 = mock_openai_response(content=None, tool_calls=[chamada])
        resposta_passo_2 = mock_openai_response(content="Entendido, houve um erro nos argumentos.")

        cliente_mock.chat.completions.create.side_effect = [resposta_passo_1, resposta_passo_2]
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        mensagens = [{"role": "user", "content": "Faça uma conta"}]
        # Não deve levantar exceção não tratada
        resposta = agent.responder(mensagens)
        assert resposta == "Entendido, houve um erro nos argumentos."

    def test_atingimento_do_limite_maximo_de_iteracoes(self, mock_openai_response, monkeypatch):
        """Garante que o loop do agente é interrompido ao atingir MAX_ITERACOES sem travar a CPU."""
        cliente_mock = MagicMock()

        # Modelo continua pedindo ferramentas infinitamente
        chamada_infinita = MockToolCall(tool_id="loop_call", name="somar", arguments={"a": 1, "b": 1})
        cliente_mock.chat.completions.create.return_value = mock_openai_response(
            content=None, tool_calls=[chamada_infinita]
        )
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)
        monkeypatch.setattr(agent, "MAX_ITERACOES", 3)

        mensagens = [{"role": "user", "content": "Loop infinito"}]
        resposta = agent.responder(mensagens)

        assert "limite de iterações" in resposta
        assert cliente_mock.chat.completions.create.call_count == 3


@pytest.mark.live_api
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-test-fake"),
    reason="Requer chave OPENAI_API_KEY real para executar Smoke Test de API viva.",
)
def test_smoke_test_api_real():
    """Smoke test opcional contra a API real da OpenAI (executado quando a chave real existe)."""
    mensagens = [
        {"role": "system", "content": "Você é um assistente conciso."},
        {"role": "user", "content": "Responda apenas com a palavra 'OK'."},
    ]
    resposta = agent.responder(mensagens)
    assert len(resposta.strip()) > 0
    assert "OK" in resposta.upper()
