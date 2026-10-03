"""
==============================================================================
TÓPICO 2: TESTES DE CONTRATO & SCHEMAS
==============================================================================

Objetivo:
Garantir que os contratos de interface entre o código Python e o modelo LLM
estejam rigorosamente alinhados com a especificação da OpenAI Tool Calling.
Um erro de digitação no schema JSON de uma ferramenta faz o modelo falhar
silenciosamente ou a API rejeitar a requisição com código 400.

O que é testado aqui:
1. Estrutura dos Schemas em `FERRAMENTAS`:
   - Cada ferramenta deve ter `"type": "function"`.
   - O objeto `"function"` deve possuir `"name"`, `"description"` e `"parameters"`.
   - `"parameters"` deve ser do tipo `"object"` e possuir `"properties"`.
2. Correspondência Bilateral (Schema <-> Código Python):
   - Toda ferramenta declarada em `FERRAMENTAS` deve ter uma função implementada
     e registrada no dicionário `EXECUTORES`.
   - Todos os parâmetros marcados como `"required"` no schema devem existir como
     argumentos válidos na assinatura da função Python correspondente (via `inspect`).
3. Contrato de Retorno do Executor:
   - A função `executar_ferramenta` deve SEMPRE retornar uma string válida em
     formato JSON, mesmo sob falhas ou exceções, pois o modelo de linguagem
     apenas consome texto.
"""

import inspect
import json
import pytest

import agent


@pytest.mark.contract
class TestContratoFerramentasSchema:
    """Valida a conformidade dos schemas de ferramentas com a especificação OpenAI."""

    def test_ferramentas_e_uma_lista_nao_vazia(self):
        """Assegura que a lista FERRAMENTAS está definida e possui elementos."""
        assert isinstance(agent.FERRAMENTAS, list)
        assert len(agent.FERRAMENTAS) > 0

    def test_estrutura_padrao_openai_function_calling(self):
        """Valida que cada ferramenta segue a especificação canônica da OpenAI."""
        for item in agent.FERRAMENTAS:
            assert isinstance(item, dict), "Cada ferramenta deve ser um dicionário"
            assert item.get("type") == "function", "O campo 'type' deve ser obrigatoriamente 'function'"

            funcao = item.get("function")
            assert isinstance(funcao, dict), "O campo 'function' deve ser um dicionário"

            nome = funcao.get("name")
            assert isinstance(nome, str) and len(nome.strip()) > 0, "Nome da função não pode ser vazio"

            descricao = funcao.get("description")
            assert isinstance(descricao, str) and len(descricao.strip()) > 0, (
                f"A função '{nome}' precisa ter uma descrição detalhada para o modelo LLM"
            )

            parametros = funcao.get("parameters")
            assert isinstance(parametros, dict), f"Parâmetros da função '{nome}' devem ser um dicionário"
            assert parametros.get("type") == "object", f"O tipo de 'parameters' em '{nome}' deve ser 'object'"
            assert "properties" in parametros, f"'{nome}' precisa ter o campo 'properties'"
            assert isinstance(parametros["properties"], dict)

    def test_correspondencia_entre_ferramentas_e_executores(self):
        """Garante que toda ferramenta declarada possui executor em Python e vice-versa."""
        nomes_declarados = {item["function"]["name"] for item in agent.FERRAMENTAS}
        nomes_executores = set(agent.EXECUTORES.keys())

        assert nomes_declarados == nomes_executores, (
            f"Divergência entre schemas declarados e executores registrados: "
            f"Declaradas={nomes_declarados}, Executores={nomes_executores}"
        )

    def test_alinhamento_de_parametros_com_assinatura_python(self):
        """Garante que os parâmetros 'required' no JSON Schema existem na assinatura Python."""
        for item in agent.FERRAMENTAS:
            nome = item["function"]["name"]
            func_python = agent.EXECUTORES[nome]
            sig = inspect.signature(func_python)
            params_python = set(sig.parameters.keys())

            props_schema = set(item["function"]["parameters"].get("properties", {}).keys())
            required_schema = set(item["function"]["parameters"].get("required", []))

            # Todos os required do schema devem existir no python
            assert required_schema.issubset(params_python), (
                f"Ferramenta '{nome}' define campos obrigatórios {required_schema - params_python} "
                f"que não constam na função Python: {params_python}"
            )

            # Todas as properties devem ser aceitas pela assinatura Python
            assert props_schema.issubset(params_python), (
                f"Ferramenta '{nome}' possui propriedades no schema que o código Python não aceita: "
                f"{props_schema - params_python}"
            )


@pytest.mark.contract
class TestContratoDeSaida:
    """Valida o contrato de saída do executor de ferramentas."""

    def test_retorno_sempre_string_json_valida(self):
        """O retorno de executar_ferramenta deve ser sempre uma string parseável por json.loads."""
        # Caso 1: Sucesso
        saida_sucesso = agent.executar_ferramenta("somar", {"a": 1, "b": 2})
        assert isinstance(saida_sucesso, str)
        parsed = json.loads(saida_sucesso)
        assert isinstance(parsed, dict)

        # Caso 2: Falha por argumentos incorretos
        saida_erro = agent.executar_ferramenta("somar", {"invalido": True})
        assert isinstance(saida_erro, str)
        parsed_erro = json.loads(saida_erro)
        assert isinstance(parsed_erro, dict)
        assert "erro" in parsed_erro
