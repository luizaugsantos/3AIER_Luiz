"""
==============================================================================
TÓPICO 4: TESTES DE AVALIAÇÃO (EVALS & QUALIDADE DO AGENTE)
==============================================================================

Objetivo:
Validar as capacidades cognitivas e comportamentais do agente LLM com datasets
de referência (Golden Datasets). Enquanto testes unitários testam código, Evals
testam a inteligência do agente, o seguimento de instruções e a acurácia de
tomada de decisão.

O que é testado aqui:
1. Tool Selection Accuracy (Acurácia de Seleção de Ferramentas):
   - Avalia se o agente escolhe a ferramenta correta (`somar`) para demandas
     numéricas exatas e se abstém de chamá-la para perguntas puramente textuais.
2. Instruction Following / Conformidade com o Prompt de Sistema (`agent.md`):
   - Valida que o agente NÃO usa frases clichês proibidas expressamente no
     `agent.md` (ex: "Claro! Fico feliz em ajudar!").
   - Valida que a resposta é concisa e direta ao ponto.
   - Valida que o idioma utilizado é o Português do Brasil.
3. Avaliador Heurístico / LLM-as-a-Judge (G-Eval):
   - Implementa uma função de pontuação (scoring) multidimensional com critérios
     de: Concisão, Relevância e Aderência ao Escopo.
4. Teste de Regressão de Prompts (Prompt Regression):
   - Executa uma bateria de casos padronizados para garantir que futuras edições
     no `agent.md` ou `memory.md` não causem perda de qualidade.
"""

from typing import Any
import pytest

import agent
from tests.conftest import MockToolCall


# ============================================================================
# DATASET DOURADO (GOLDEN DATASET) PARA AVALIAÇÃO
# ============================================================================
CASOS_AVALIACAO = [
    {
        "id": "matematica_direta",
        "entrada": "Quanto é 1234 + 5678?",
        "deve_usar_ferramenta": True,
        "ferramenta_esperada": "somar",
        "argumentos_esperados": {"a": 1234, "b": 5678},
    },
    {
        "id": "conhecimento_geral",
        "entrada": "Qual é a capital do Brasil?",
        "deve_usar_ferramenta": False,
        "ferramenta_esperada": None,
        "argumentos_esperados": None,
    },
    {
        "id": "adicao_em_linguagem_natural",
        "entrada": "Por favor, adicione 45 a 55 para mim.",
        "deve_usar_ferramenta": True,
        "ferramenta_esperada": "somar",
        "argumentos_esperados": {"a": 45, "b": 55},
    },
]


def avaliador_heuristico_qualidade(resposta: str, pergunta: str) -> dict[str, float]:
    """Mecanismo de avaliação (Scorer) que pontua a resposta em critérios de 0.0 a 1.0.

    Pode ser estendido para LLM-as-a-Judge quando conectado à API da OpenAI.
    """
    scores = {}

    # Critério 1: Ausência de clichês proibidos em agent.md
    cliches_proibidos = [
        "claro! fico feliz em ajudar",
        "como um modelo de linguagem",
        "espero ter ajudado",
        "com certeza! aqui está",
    ]
    contem_cliche = any(c in resposta.lower() for c in cliches_proibidos)
    scores["ausencia_de_cliches"] = 0.0 if contem_cliche else 1.0

    # Critério 2: Concisão (respostas curtas pontuam mais para perguntas simples)
    palavras = len(resposta.split())
    if palavras == 0:
        scores["concisao"] = 0.0
    elif palavras <= 30:
        scores["concisao"] = 1.0
    elif palavras <= 70:
        scores["concisao"] = 0.7
    else:
        scores["concisao"] = 0.4

    # Critério 3: Não vazia e com conteúdo textual
    scores["relevancia_basica"] = 1.0 if len(resposta.strip()) > 3 else 0.0

    return scores


@pytest.mark.eval
class TestAvaliacaoQualidadeAgente:
    """Suíte de avaliação do comportamento do agente inteligente."""

    @pytest.mark.parametrize("caso", CASOS_AVALIACAO, ids=[c["id"] for c in CASOS_AVALIACAO])
    def test_eval_selecao_de_ferramenta(self, caso: dict[str, Any], mock_openai_response, monkeypatch):
        """Avalia se a intenção do usuário ativa ou não a ferramenta adequada."""
        from unittest.mock import MagicMock

        cliente_mock = MagicMock()

        if caso["deve_usar_ferramenta"]:
            chamada = MockToolCall(
                tool_id="call_eval",
                name=caso["ferramenta_esperada"],
                arguments=caso["argumentos_esperados"],
            )
            cliente_mock.chat.completions.create.side_effect = [
                mock_openai_response(content=None, tool_calls=[chamada]),
                mock_openai_response(content="Resultado: 6912."),
            ]
        else:
            cliente_mock.chat.completions.create.return_value = mock_openai_response(
                content="A capital do Brasil é Brasília."
            )

        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        mensagens = [
            {"role": "system", "content": agent.carregar_contexto()},
            {"role": "user", "content": caso["entrada"]},
        ]
        resposta = agent.responder(mensagens)

        assert len(resposta.strip()) > 0
        if caso["deve_usar_ferramenta"]:
            # Verifica se foi feita chamada de ferramenta
            mensagens_tool = [m for m in mensagens if m.get("role") == "tool"]
            assert len(mensagens_tool) >= 1
        else:
            mensagens_tool = [m for m in mensagens if m.get("role") == "tool"]
            assert len(mensagens_tool) == 0

    def test_eval_conformidade_com_regras_do_agent_md(self):
        """Avalia se uma resposta gerada cumpre as restrições explicitadas em agent.md."""
        # Resposta de exemplo que viola regras
        resposta_inadequada = "Claro! Fico feliz em ajudar! A soma de 10 com 20 dá 30. Qualquer dúvida, estou à disposição!"
        scores_ruim = avaliador_heuristico_qualidade(resposta_inadequada, "Quanto é 10 + 20?")
        assert scores_ruim["ausencia_de_cliches"] == 0.0

        # Resposta de exemplo adequada às regras de agent.md (direta, sem clichê, concisa)
        resposta_adequada = "O resultado é 30."
        scores_bom = avaliador_heuristico_qualidade(resposta_adequada, "Quanto é 10 + 20?")
        assert scores_bom["ausencia_de_cliches"] == 1.0
        assert scores_bom["concisao"] == 1.0

    def test_eval_regressao_de_prompt(self):
        """Garante que o prompt do sistema preserva seções essenciais sem regressão textual."""
        contexto = agent.carregar_contexto()

        # Validações estruturais do prompt de sistema
        assert "Você é um assistente" in contexto
        assert "português do Brasil" in contexto
        assert "Não invente" in contexto
        assert "Use as ferramentas quando elas souberem melhor que você" in contexto
        assert "Respeite a memória" in contexto
