"""
==============================================================================
TÓPICO 7: TESTES DE INTERFACE DO USUÁRIO (UI STREAMLIT)
==============================================================================

Objetivo:
Validar a interface gráfica do usuário (`app.py`) de forma automatizada e
sem necessidade de abrir um navegador web. O Streamlit fornece o módulo
`streamlit.testing.v1.AppTest`, que simula a árvore de renderização (DOM virtual),
interações do usuário e o estado da sessão (`st.session_state`).

O que é testado aqui:
1. Comportamento com Chave Ausente:
   - Se `OPENAI_API_KEY` não estiver definida, o app deve exibir mensagem de erro
     amigável (`st.error`) e interromper o script (`st.stop`) sem gerar stack trace.
2. Renderização Inicial dos Componentes:
   - Validação da presença do título `st.title("🤖 Agente base")`.
   - Validação da sidebar com a legenda indicando o modelo em uso (`agent.MODELO`).
   - Validação da presença do botão "Nova conversa" e do campo de chat `st.chat_input`.
3. Interação do Usuário via Chat:
   - Simulação de digitação no `chat_input`.
   - Atualização automática do histórico visível (`st.session_state.tela`) e do
     histórico estruturado (`st.session_state.mensagens`).
4. Ação do Botão "Nova Conversa":
   - Clicar no botão limpa a tela e reinicia o histórico apenas com a mensagem
     de sistema original.
5. Tratamento de Exceções no Chat:
   - Se o modelo falhar durante a chamada, o app exibe `st.error` e reverte o
     turno incompleto sem corromper as mensagens subsequentes.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Garante skip gracioso caso streamlit não esteja instalado no ambiente de execução
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
import agent  # noqa: E402

CAMINHO_APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.mark.ui
class TestInterfaceStreamlit:
    """Testes automatizados da UI Streamlit em app.py usando AppTest."""

    def test_tela_exibe_erro_se_api_key_estiver_ausente(self, monkeypatch):
        """Garante exibição amigável de erro na tela quando OPENAI_API_KEY não existe."""
        monkeypatch.setattr(agent, "_cliente", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        at = AppTest.from_file(str(CAMINHO_APP), default_timeout=10)
        at.run()

        # Deve exibir erro de chave ausente e interromper execução
        assert len(at.error) > 0
        assert "OPENAI_API_KEY não encontrada" in at.error[0].value

    def test_renderizacao_inicial_com_chave_configurada(self, monkeypatch):
        """Valida que todos os componentes principais da interface são renderizados."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-para-interface-teste-123")
        monkeypatch.setattr(agent, "_cliente", MagicMock())

        at = AppTest.from_file(str(CAMINHO_APP), default_timeout=10)
        at.run()

        # Valida título
        assert len(at.title) > 0
        assert "🤖 Agente base" in at.title[0].value

        # Valida componentes da barra lateral
        assert len(at.sidebar.caption) > 0
        assert agent.MODELO in at.sidebar.caption[0].value
        assert len(at.sidebar.button) > 0
        assert at.sidebar.button[0].label == "Nova conversa"

        # Valida campo de input do usuário
        assert len(at.chat_input) > 0

    def test_envio_de_mensagem_do_usuario_e_resposta(self, monkeypatch):
        """Simula o envio de mensagem pelo usuário e exibição da resposta do agente."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-para-interface-teste-123")
        monkeypatch.setattr(agent, "_cliente", MagicMock())

        # Mock da resposta do agente para evitar chamadas de rede
        resposta_simulada = "Olá! Como posso te ajudar?"
        monkeypatch.setattr(agent, "responder", lambda msgs: resposta_simulada)

        at = AppTest.from_file(str(CAMINHO_APP), default_timeout=10)
        at.run()

        # Usuário envia mensagem
        at.chat_input[0].set_value("Olá, agente!").run()

        # Valida que o histórico de tela contém a mensagem do usuário e da IA
        tela = at.session_state["tela"]
        assert len(tela) == 2
        assert tela[0] == ("user", "Olá, agente!")
        assert tela[1] == ("assistant", resposta_simulada)

    def test_botao_nova_conversa_reseta_estado(self, monkeypatch):
        """Valida se o botão 'Nova conversa' zera as mensagens da tela e restaura o sistema."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-para-interface-teste-123")
        monkeypatch.setattr(agent, "_cliente", MagicMock())
        monkeypatch.setattr(agent, "responder", lambda msgs: "Resposta qualquer")

        at = AppTest.from_file(str(CAMINHO_APP), default_timeout=10)
        at.run()

        # Usuário envia uma pergunta inicial
        at.chat_input[0].set_value("Mensagem teste").run()
        assert len(at.session_state["tela"]) > 0

        # Usuário clica no botão "Nova conversa"
        at.sidebar.button[0].click().run()

        # O estado deve ter sido resetado
        assert at.session_state["tela"] == []
        assert len(at.session_state["mensagens"]) == 1
        assert at.session_state["mensagens"][0]["role"] == "system"

    def test_tratamento_de_erro_no_chat_sem_corromper_historico(self, monkeypatch):
        """Garante que falhas de chamada ao modelo exibem erro na tela e revertem o turno incompleto."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-para-interface-teste-123")
        monkeypatch.setattr(agent, "_cliente", MagicMock())

        # Simula erro de conexão ao chamar o modelo
        def erro_chamada(msgs):
            raise ConnectionError("Falha de conexão com a OpenAI")

        monkeypatch.setattr(agent, "responder", erro_chamada)

        at = AppTest.from_file(str(CAMINHO_APP), default_timeout=10)
        at.run()

        # Usuário envia mensagem que causará erro
        at.chat_input[0].set_value("Pergunta que falhará").run()

        # Deve exibir componente st.error informando o problema
        assert len(at.error) > 0
        assert "Falha de conexão com a OpenAI" in at.error[0].value

        # O histórico reverte o que o assistente tentou adicionar, mantendo a pergunta
        mensagens = at.session_state["mensagens"]
        assert len(mensagens) == 2  # system + user
        assert mensagens[0]["role"] == "system"
        assert mensagens[1]["role"] == "user"
        assert mensagens[1]["content"] == "Pergunta que falhará"
