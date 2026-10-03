"""
==============================================================================
TÓPICO 6: TESTES DE MEMÓRIA E ESTADO
==============================================================================

Objetivo:
Validar o gerenciamento do estado conversacional e a persistência de contexto
de longo prazo via `memory.md`. Em agentes autônomos, a retenção de fatos sobre
o usuário e a integridade do histórico entre turnos são fundamentais para
conversas coerentes e personalizadas.

O que é testado aqui:
1. Injeção e Prioridade da Memória no Prompt de Sistema:
   - Garante que a seção `# Memória` é inserida e que a instrução explícita
     de prioridade sobre suposições é comunicada ao modelo.
2. Atualização Dinâmica da Memória (Persistência):
   - Simula a gravação de novos fatos ou preferências no arquivo `memory.md` e
     assegura que `carregar_contexto()` reflete imediatamente as mudanças
     sem necessidade de reiniciar o processo Python.
3. Preservação de Estado entre Múltiplos Turnos:
   - Valida que uma conversa com várias trocas de mensagens (Turno 1, Turno 2...)
     mantém todo o histórico linear acumulado na lista `mensagens`.
4. Monitoramento e Proteção contra Context Window Overflow:
   - Mede o volume de caracteres acumulados e testa uma função utilitária de
     estimativa de tokens para prevenir que a conversa estoure o limite de
     contexto do modelo (`gpt-4o-mini`).
"""

from unittest.mock import MagicMock
import pytest

import agent


def estimar_tokens_aproximados(mensagens: list[dict]) -> int:
    """Estimativa rápida baseada em caracteres (~4 caracteres por token em média)."""
    total_chars = sum(len(str(m.get("content", ""))) for m in mensagens)
    return total_chars // 4


@pytest.mark.memory
class TestMemoriaEEstado:
    """Suíte de validação de estado, persistência e janela de contexto."""

    def test_injecao_da_memoria_no_contexto_do_sistema(self):
        """Verifica se a memória é corretamente injetada e formatada no prompt de sistema."""
        contexto = agent.carregar_contexto()
        assert "# Memória" in contexto
        assert "Respeite a memória" in contexto

    def test_atualizacao_dinamica_de_fatos_na_memoria(self, tmp_path, monkeypatch):
        """Garante que alterações em disco no arquivo de memória surtem efeito imediato."""
        arquivo_temp_memory = tmp_path / "temp_memory.md"
        arquivo_temp_memory.write_text("# Memória\n\n- Nome: Carlos\n", encoding="utf-8")

        monkeypatch.setattr(agent, "MEMORY_MD", arquivo_temp_memory)

        # Leitura inicial
        contexto_1 = agent.carregar_contexto()
        assert "Nome: Carlos" in contexto_1

        # Atualização dinâmica de memória
        arquivo_temp_memory.write_text(
            "# Memória\n\n- Nome: Carlos\n- Prefere respostas em tópicos.\n",
            encoding="utf-8",
        )

        # Leitura subsequente deve conter o novo fato sem reiniciar processo
        contexto_2 = agent.carregar_contexto()
        assert "Prefere respostas em tópicos." in contexto_2

    def test_preservacao_de_estado_em_multiplos_turnos(self, mock_openai_response, monkeypatch):
        """Valida que o histórico de mensagens acumula os turnos sequencialmente."""
        cliente_mock = MagicMock()
        cliente_mock.chat.completions.create.side_effect = [
            mock_openai_response(content="Olá, Carlos! Tudo bem?"),
            mock_openai_response(content="Você me disse que prefere respostas em tópicos."),
        ]
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        # Inicializa com prompt de sistema
        mensagens = [{"role": "system", "content": agent.carregar_contexto()}]

        # Turno 1
        mensagens.append({"role": "user", "content": "Meu nome é Carlos."})
        resposta_1 = agent.responder(mensagens)
        assert resposta_1 == "Olá, Carlos! Tudo bem?"
        assert len(mensagens) == 3

        # Turno 2
        mensagens.append({"role": "user", "content": "O que você sabe sobre mim?"})
        resposta_2 = agent.responder(mensagens)
        assert resposta_2 == "Você me disse que prefere respostas em tópicos."
        assert len(mensagens) == 5

        # Sequência esperada: system -> user -> assistant -> user -> assistant
        papeis = [m["role"] for m in mensagens]
        assert papeis == ["system", "user", "assistant", "user", "assistant"]

    def test_verificacao_de_limite_de_contexto(self, mock_openai_response, monkeypatch):
        """Monitora o crescimento de tokens para prevenir overflow da janela de contexto."""
        cliente_mock = MagicMock()
        cliente_mock.chat.completions.create.return_value = mock_openai_response(
            content="Mensagem confirmada."
        )
        monkeypatch.setattr(agent, "get_cliente", lambda: cliente_mock)

        mensagens = [{"role": "system", "content": agent.carregar_contexto()}]

        # Simula 20 turnos de conversação
        for i in range(20):
            mensagens.append({"role": "user", "content": f"Pergunta número {i} sobre o projeto."})
            agent.responder(mensagens)

        # O modelo gpt-4o-mini possui janela de 128k tokens; garantimos que histórico normal fica dentro do esperado
        tokens_estimados = estimar_tokens_aproximados(mensagens)
        assert tokens_estimados < 10000, f"Histórico muito grande: {tokens_estimados} tokens estimados"
        assert len(mensagens) == 1 + (20 * 2)  # system + 20 pares user/assistant
