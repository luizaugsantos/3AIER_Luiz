# Deploy na AWS (EC2 + GitHub Actions)

Arquitetura: o GitHub Actions roda os testes, empacota o app, envia o zip para um
bucket S3 e pede à EC2 (via SSM) para aplicá-lo. O Streamlit roda como serviço
systemd na porta 80. A chave do modelo fica no Secrets Manager. O GitHub
autentica na AWS por OIDC (sem chaves de acesso).

## 1. Antes de começar
1. Faça um **fork** do repositório e aponte seu `origin` para ele.
2. Conta AWS ativa, região escolhida (ex.: `us-east-1`) e VPC padrão existente.
3. Uma chave de API (Gemini ou OpenAI). Não cole em chat nem no código.

## 2. Criar a infraestrutura
CloudFormation > Create stack > Upload template: `infra/aws/agente-streamlit.yml`.

| Parâmetro | Valor |
|---|---|
| Stack name | `agente-streamlit` |
| GitHubRepo | `seu-usuario/3AIER_Luiz` |
| GitHubBranch | `main` |
| LlmApiKey | sua chave |
| LlmBaseUrl / LlmModel | Gemini: valores padrão. OpenAI: base URL vazia e `gpt-4o-mini` |
| CreateGitHubOidcProvider | `true`; se der erro de "já existe", recrie com `false` |

Marque o reconhecimento de IAM e crie. Aguarde `CREATE_COMPLETE` (5 a 10 min).

## 3. Configurar o GitHub
Repositório > Settings > Secrets and variables > Actions > **Variables**, com os
valores da aba Outputs da stack:
`AWS_ROLE_ARN` (GitHubRoleArn), `AWS_REGION` (Region), `AWS_STACK_NAME` (StackName).

## 4. Deploy
Faça push na `main` ou rode manualmente Actions > "Deploy AWS" > Run workflow.
Jobs: **test** (lint + pytest) → **deploy** → **smoke-test**.
Abra `AppUrl` (output da stack) no navegador e converse com o agente.

## 5. Evidências para a entrega
- Print do console AWS: CloudFormation (Resources) e/ou EC2 com a instância rodando.
- Print do navegador com o front respondendo.
- Print da Action verde (opcional).
- Documento com nomes, link do repositório e `PROMPTS_USADOS.md`.

## 6. Problemas comuns
| Sintoma | Causa |
|---|---|
| Deploy "pulado" | Faltam as Variables do GitHub |
| Erro ao assumir a role | `GitHubRepo`/branch diferentes do repositório real |
| Página abre mas agente dá erro de chave | Chave/base URL/modelo errados no secret; atualize o secret e rode o deploy de novo |
| SSM não aceita comando | Instância ainda iniciando; o workflow tenta por ~10 min |
| Falha ao criar provedor OIDC | Já existe na conta: use `CreateGitHubOidcProvider=false` |

## 7. Limpeza (para não gerar custo)
1. Esvazie o bucket S3 (o CloudFormation não apaga bucket com objetos).
2. Delete a stack. O Elastic IP e a EC2 geram custo enquanto existirem.
