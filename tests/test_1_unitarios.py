"""
==============================================================================
TÓPICO 1: TESTES UNITÁRIOS (CÓDIGO PYTHON LOCAL)
==============================================================================

Objetivo:
Validar as funções determinísticas e puramente locais do agente isoladas de
qualquer modelo de linguagem ou chamada de rede externa.

O que é testado aqui:
1. Funções locais de ferramentas (`somar`) com entradas variadas (positivos,
   negativos, decimais e zeros).
2. O executor seguro de ferramentas (`executar_ferramenta`), garantindo que
   erros de execução ou chamadas a ferramentas inexistentes nunca derrubem o
   processo do agente e sempre devolvam um JSON estruturado.
3. Carregamento do contexto (`carregar_contexto`), assegurando a leitura e
   composição correta dos arquivos `agent.md` e `memory.md`, bem como o
   comportamento de fallback caso os arquivos não existam.
4. Inicialização do cliente OpenAI (`get_cliente`), verificando tanto o
   lançamento de exceção amigável quando a chave está ausente quanto o padrão
   singleton quando configurada.
5. Verificação dos valores de configuração padrão (`MODELO`, `TEMPERATURA`,
   `MAX_ITERACOES`).
"""

import json
import pytest

import agent


@pytest.mark.unit
class TestFuncoesFerramentas:
    """Testes unitários para as funções Python que servem como ferramentas do agente."""

    def test_somar_com_numeros_positivos(self):
        """Valida que somar números inteiros positivos retorna o resultado correto."""
        resultado = agent.somar(a=10, b=25)
        assert resultado == {"resultado": 35}

    def test_somar_com_numeros_negativos_e_decimais(self):
        """Valida precisão com números negativos e números de ponto flutuante."""
        resultado = agent.somar(a=-5.5, b=2.2)
        assert pytest.approx(resultado["resultado"], rel=1e-5) == -3.3

    def test_somar_com_zero(self):
        """Valida elemento neutro da adição."""
        resultado = agent.somar(a=0, b=0)
        assert resultado == {"resultado": 0}


@pytest.mark.unit
class TestExecutarFerramenta:
    """Testes para o despachante e tratador de execução de ferramentas."""

    def test_executar_ferramenta_sucesso(self):
        """Valida que uma ferramenta registrada é executada e retorna string JSON."""
        retorno_str = agent.executar_ferramenta("somar", {"a": 7, "b": 3})
        dados = json.loads(retorno_str)
        assert dados == {"resultado": 10}

    def test_executar_ferramenta_inexistente(self):
        """Garante que ferramenta inexistente devolve mensagem de erro amigável sem crash."""
        retorno_str = agent.executar_ferramenta("multiplicar", {"a": 2, "b": 3})
        dados = json.loads(retorno_str)
        assert "erro" in dados
        assert "não existe" in dados["erro"]

    def test_executar_ferramenta_com_argumentos_invalidos(self):
        """Garante que erro nos argumentos (TypeError) é capturado e devolvido como JSON."""
        retorno_str = agent.executar_ferramenta("somar", {"parametro_invalido": 123})
        dados = json.loads(retorno_str)
        assert "erro" in dados
        assert "Falha em 'somar'" in dados["erro"]


@pytest.mark.unit
class TestCarregamentoDeContexto:
    """Testes para leitura de prompts de sistema e arquivos de contexto."""

    def test_carregar_contexto_com_arquivos_reais(self):
        """Garante que agent.md e memory.md são concatenados com o separador esperado."""
        contexto = agent.carregar_contexto()
        assert isinstance(contexto, str)
        assert "# Memória" in contexto
        assert "---" in contexto

    def test_carregar_contexto_quando_arquivos_nao_existem(self, monkeypatch, tmp_path):
        """Valida comportamento de fallback caso os arquivos markdown sejam deletados."""
        caminho_inexistente_agent = tmp_path / "inexistente_agent.md"
        caminho_inexistente_memory = tmp_path / "inexistente_memory.md"

        monkeypatch.setattr(agent, "AGENT_MD", caminho_inexistente_agent)
        monkeypatch.setattr(agent, "MEMORY_MD", caminho_inexistente_memory)

        contexto = agent.carregar_contexto()
        assert "Você é um assistente." in contexto


@pytest.mark.unit
class TestConfiguracaoECliente:
    """Testes para inicialização e ciclo de vida do cliente OpenAI."""

    def test_get_cliente_sem_chave_lanca_runtime_error(self, monkeypatch):
        """Garante erro explicativo quando OPENAI_API_KEY não foi configurada."""
        monkeypatch.setattr(agent, "_cliente", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY não encontrada"):
            agent.get_cliente()

    def test_get_cliente_singleton(self, monkeypatch):
        """Garante que get_cliente() reutiliza a mesma instância após instanciado."""
        monkeypatch.setattr(agent, "_cliente", None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-para-teste")

        cliente1 = agent.get_cliente()
        cliente2 = agent.get_cliente()
        assert cliente1 is cliente2

        # Limpeza para evitar efeitos colaterais
        monkeypatch.setattr(agent, "_cliente", None)

    def test_valores_padrao_de_configuracao(self):
        """Valida se as constantes de configuração possuem tipos e valores adequados."""
        assert isinstance(agent.MODELO, str)
        assert isinstance(agent.TEMPERATURA, float)
        assert isinstance(agent.MAX_ITERACOES, int)
        assert 0.0 <= agent.TEMPERATURA <= 2.0
        assert agent.MAX_ITERACOES > 0
