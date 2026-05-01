# Datathon FIAP — Previsão do próximo fechamento da ação da Nike com LSTM, MLflow e FastAPI

## Arquitetura do projeto
![Arquitetura do projeto](/Datathon-FIAP/DatathonArq.png)


## Visão geral

Este projeto foi desenvolvido no contexto do Datathon da pós-graduação da FIAP com foco em MLOps e Engenharia de Machine Learning.

A solução tem como objetivo prever o comportamento de curto prazo da ação da Nike (`NKE`) a partir de uma série temporal histórica obtida via Yahoo Finance. O projeto evoluiu de um experimento inicial de modelagem para uma estrutura mais próxima de produção, com pipeline de dados, treinamento rastreável, versionamento de modelo, inferência via API e componentes de observabilidade.

No estado atual, o projeto já contempla:

- coleta automática de dados históricos com `yfinance`
- separação entre dados brutos e processados
- engenharia de atributos para séries temporais
- treinamento de modelo LSTM em PyTorch
- rastreamento de experimentos com MLflow
- versionamento de modelos via MLflow Model Registry
- inferência desacoplada de artefatos locais fixos
- API em FastAPI para serving
- orquestração local com Docker Compose
- versionamento de pipeline e dados com DVC
- estrutura para monitoramento com Prometheus e Grafana
- preparação de ambiente com Airflow para orquestração
- explicação de previsões com LLM local via Ollama
- RAG com ChromaDB para perguntas sobre o projeto
- servidor MCP para integração com Claude Code e clientes compatíveis

---

## Objetivo do projeto

O objetivo principal é prever o **retorno do próximo dia** da ação da Nike e, a partir dele, reconstruir o **preço previsto de fechamento**.

Em vez de prever diretamente o valor absoluto do preço, a estratégia adotada foi prever o retorno diário futuro (`target_return_1d`). Essa abordagem tende a deixar o problema mais estável do ponto de vista estatístico e reduz o risco de previsões excessivamente suavizadas, algo comum quando se tenta prever preço absoluto diretamente em séries financeiras.

### Entrada do problema

O pipeline utiliza histórico de mercado da Nike com colunas como:

- data
- open
- high
- low
- close
- volume

### Saída do problema

O modelo retorna:

- `predicted_return`: retorno previsto para o próximo dia
- `predicted_price`: preço de fechamento estimado para o próximo dia

---

## Arquitetura da solução

O projeto está organizado como uma pipeline ponta a ponta de ML.

### 1. Coleta dos dados

Os dados históricos da Nike são baixados com `yfinance`, usando uma janela temporal configurável em meses.

Responsabilidades dessa etapa:

- definir o ticker
- buscar histórico ajustado
- salvar os dados brutos em `data/raw`

### 2. Processamento e engenharia de atributos

Após a coleta, os dados passam por um pré-processamento que:

- padroniza nomes de colunas
- converte tipos numéricos
- ordena por data
- remove inconsistências
- cria features derivadas para alimentar o modelo

As principais features atualmente utilizadas são:

- `close`
- `return_1d`
- `ma_5_ratio`
- `ma_20_ratio`
- `volatility_10`
- `volume_zscore_20`

Também é criado o alvo:

- `target_return_1d`

### 3. Criação das sequências temporais

Como o modelo é uma LSTM, os dados são transformados em janelas temporais com tamanho configurável (`seq_length`).

Nessa etapa também ocorre:

- split temporal entre treino, validação e teste
- ajuste do `MinMaxScaler` apenas no conjunto de treino
- geração dos artefatos necessários para reuso em inferência:
  - `scaler`
  - `feature_cols`
  - `seq_length`

### 4. Treinamento do modelo

O modelo foi implementado em **PyTorch**.

Arquitetura atual:

- 1ª camada LSTM bidirecional com 64 unidades
- Dropout
- 2ª camada LSTM com 32 unidades
- camadas densas intermediárias
- camada final de regressão com saída escalar

Estratégias de treino:

- `HuberLoss`
- `Adam`
- `ReduceLROnPlateau`
- `EarlyStopping`

### 5. Avaliação

Depois do treino, o pipeline gera previsões no conjunto de teste e compara:

- preço real
- preço previsto
- baseline ingênuo

As métricas principais são:

- `RMSE`
- `MAE`
- `MAPE`
- `Direction Accuracy`
- comparação com baseline por `naive_rmse` e `naive_mae`

### 6. Tracking e versionamento com MLflow

O projeto utiliza MLflow para:

- registrar parâmetros de treino
- registrar métricas
- salvar gráficos
- armazenar artefatos do run
- logar o modelo PyTorch
- salvar metadados do pré-processamento
- registrar novas versões do modelo no Model Registry
- associar aliases como `candidate` e `champion`

### 7. Serving e inferência

A inferência não depende mais de um `model.pkl`, `scaler.pkl` ou arquivo local fixo como fonte principal.

O fluxo atual utiliza:

- carregamento do modelo a partir do Model Registry
- recuperação do metadata do pré-processamento
- reconstrução da janela de entrada
- geração do retorno e do preço previsto

Além da inferência por script, o projeto já possui uma API FastAPI para exposição do modelo.

### 8. LLM, RAG e MCP

Além do serving tradicional do modelo, o projeto também possui uma camada de inteligência generativa para melhorar a experiência de uso e explicabilidade da solução. Essa camada utiliza **Ollama** para execução local de modelos de linguagem, **ChromaDB** como vector store para RAG e um **servidor MCP** para integração com clientes compatíveis, como Claude Code.

Responsabilidades dessa etapa:

- gerar explicações em linguagem natural para as previsões do modelo LSTM
- responder perguntas sobre o próprio projeto com base em documentos indexados
- expor ferramentas MCP para previsão, consulta de metadados, explicação e perguntas via RAG
- manter a stack de LLM/RAG desacoplada da stack principal por meio de um `docker-compose.llm.yaml`

O fluxo geral dessa camada é:

1. A API recebe uma requisição de previsão, explicação ou pergunta.
2. Para previsão, a FastAPI chama o modelo LSTM registrado no MLflow Model Registry.
3. Para explicação, o retorno da previsão é enviado ao Ollama para geração de texto em linguagem natural.
4. Para perguntas sobre o projeto, documentos como `README.md` e `params.yaml` são indexados no ChromaDB.
5. Os chunks mais relevantes são recuperados e enviados como contexto para a LLM gerar a resposta.

### 9. Observabilidade e operação

O repositório também inclui componentes de suporte operacional:

- `docker-compose.yaml`
- pasta `prometheus`
- pasta `grafana/provisioning`
- workflows em `.github/workflows`
- estrutura de testes em `tests`
- uso de `logs/pipeline.log` para rastreabilidade local
- estrutura com Airflow para orquestração

---

## Estrutura do projeto

Abaixo está a estrutura lógica do projeto com os principais diretórios e scripts utilizados na solução:

```text
Datathon-FIAP/
├── .dvc/                               # Metadados internos do DVC
├── .claude/
│   └── mcp.json                        # Configuração do servidor MCP para clientes compatíveis
├── .github/
│   └── workflows/
│       └── tests.yml                   # Workflow de CI para execução automatizada de testes
├── data/
│   ├── raw/                            # Dados brutos coletados
│   └── processed/                      # Dados tratados para treino
├── docs/
│   └── architecture_llm_rag_mcp.md      # Diagrama e detalhamento da arquitetura LLM/RAG/MCP
├── fast_api/
│   └── main.py                         # Arquivo alternativo/auxiliar para inicialização da API
├── grafana/
│   └── provisioning/
│       ├── dashboards/                 # Provisionamento automático de dashboards do Grafana
│       └── datasources/                # Provisionamento automático de data sources do Grafana
├── logs/                               # Logs gerados pela aplicação
├── notebooks/                          # Estudos, EDA e experimentos
├── outputs/                            # Artefatos locais gerados pelo treino
├── prometheus/                         # Configurações de monitoramento
├── src/
│   ├── api/
│   │   ├── app.py                      # Aplicação principal da FastAPI
│   │   ├── schema.py                   # Schemas Pydantic de request/response da API
│   │   └── service.py                  # Regras de negócio da API e chamada de inferência
│   ├── inference/
│   │   └── inference.py                # Inferência local usando modelo do registry
│   ├── data_loader.py                  # Download, leitura e persistência de dados brutos
│   ├── logger_config.py                # Configuração central de logging
│   ├── main.py                         # Pipeline principal de treino, avaliação e registro no MLflow
│   ├── mlflow_utils.py                 # Configuração de tracking e registry no MLflow
│   ├── model_registry.py               # Registro, alias e carregamento de modelos no MLflow
│   ├── mcp_server.py                   # Servidor MCP para integração com clientes externos
│   ├── prepare_data.py                 # Pipeline de preparação dos dados raw e processed
│   ├── preprocessing.py                # Tratamento dos dados e criação das features/sequências
│   └── train.py                        # Arquitetura LSTM, treino, avaliação e predição
├── tests/
│   ├── __init__.py                     # Marca o diretório de testes como pacote Python
│   ├── conftest.py                     # Fixtures e configurações compartilhadas de testes
│   ├── test_api_endpoints.py           # Testes dos endpoints expostos pela API
│   ├── test_api_schema.py              # Testes de validação dos schemas Pydantic
│   └── test_api_service.py             # Testes da camada de serviço da API
├── .dvcignore                          # Arquivos ignorados pelo DVC
├── .gitignore                          # Arquivos ignorados pelo Git
├── Dockerfile                          # Construção da imagem da aplicação
├── README.md                           # Documentação principal do projeto
├── README_mlops.md                     # Versão/documentação complementar focada em MLOps
├── docker-compose.yaml                 # Orquestração dos serviços containerizados principais
├── docker-compose.llm.yaml             # Stack adicional para Ollama e ChromaDB
├── dvc.yaml                            # Pipeline versionada com DVC
├── env_example                         # Exemplo de variáveis de ambiente
├── params.yaml                         # Parâmetros centrais da pipeline
├── pyproject.toml                      # Dependências e configuração do projeto
```


## Camada de LLM, RAG e MCP

A branch de LLM adiciona uma camada complementar à API de previsão. O objetivo não é substituir o modelo LSTM, mas enriquecer o projeto com explicações em linguagem natural, perguntas sobre a documentação e integração com ferramentas externas via MCP.

### Explicação de previsões com Ollama

O endpoint `/explain` recebe o último preço de fechamento conhecido, executa a previsão com o modelo LSTM e envia o resultado para uma LLM local via Ollama. A resposta final combina os valores previstos com uma explicação textual mais amigável.

Exemplo de chamada:

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"close": 95.5}'
```

Exemplo de resposta:

```json
{
  "explanation": "O modelo prevê alta moderada para a Nike amanhã...",
  "close": 95.5,
  "predicted_price": 95.9,
  "predicted_return": 0.0042
}
```

Variáveis de ambiente relacionadas:

| Variável | Valor padrão | Descrição |
|---|---:|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do servidor Ollama |
| `OLLAMA_MODEL` | `llama3.2:1b` | Modelo usado para geração de texto |
| `OLLAMA_TIMEOUT` | `30` | Timeout das chamadas ao Ollama, em segundos |

### RAG para perguntas sobre o projeto

O projeto também implementa RAG com **ChromaDB** e **Ollama**. Essa abordagem permite fazer perguntas sobre a própria documentação do projeto, usando os documentos indexados como contexto para a resposta da LLM.

Funcionamento resumido:

1. Documentos do projeto, como `README.md` e `params.yaml`, são indexados no ChromaDB.
2. Os embeddings são gerados pelo modelo `nomic-embed-text` via Ollama.
3. Ao receber uma pergunta, o sistema busca os chunks mais relevantes no vector store.
4. Os chunks recuperados são enviados como contexto para a LLM gerar a resposta.

Exemplo de chamada ao endpoint `/chat`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o seq_length do modelo?"}'
```

Exemplo de resposta:

```json
{
  "answer": "O modelo usa seq_length de 60 dias...",
  "question": "Qual o seq_length do modelo?"
}
```

Variáveis de ambiente relacionadas:

| Variável | Valor padrão | Descrição |
|---|---:|---|
| `CHROMA_HOST` | `localhost` | Host do ChromaDB |
| `CHROMA_PORT` | `8100` | Porta do ChromaDB |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Modelo usado para geração de embeddings |

### Servidor MCP

O projeto inclui um servidor **MCP (Model Context Protocol)** para permitir que clientes compatíveis interajam com a API de previsão. Na prática, o MCP funciona como uma camada de ferramentas que chama a FastAPI por HTTP.

Ferramentas disponíveis:

| Tool | Descrição |
|---|---|
| `predict(close)` | Prevê o próximo preço de fechamento |
| `model_info()` | Retorna informações do modelo registrado |
| `explain_prediction(close)` | Gera explicação em linguagem natural para uma previsão |
| `ask_about_model(question)` | Responde perguntas sobre o projeto usando RAG |

O arquivo `.claude/mcp.json` já pode ser usado para configuração com Claude Code. Para configurar manualmente, utilize uma estrutura semelhante:

```json
{
  "mcpServers": {
    "nike-forecast": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "FASTAPI_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Arquitetura da camada LLM/RAG/MCP

O diagrama completo com fluxos detalhados está em `docs/architecture_llm_rag_mcp.md`.

```text
┌──────────────────┐
│   Claude Code    │
│  (MCP Client)    │
└────────┬─────────┘
         │ MCP (stdio)
         ▼
┌──────────────────┐       ┌────────────┐
│   MCP Server     │──────▶│  FastAPI    │
│ predict          │ HTTP  │ /predict   │
│ model_info       │       │ /explain   │
│ explain_prediction│      │ /chat      │
│ ask_about_model  │       │ /model-info│
└──────────────────┘       └──┬─────┬───┘
                              │     │
                    ┌─────────┘     └─────────┐
                    ▼                         ▼
              ┌───────────┐           ┌─────────────┐
              │  MLflow   │           │ LLM Service │
              │ Registry  │           │             │
              └───────────┘           └──────┬──────┘
                                             │
                                    ┌────────┴────────┐
                                    ▼                 ▼
                              ┌──────────┐     ┌───────────┐
                              │  Ollama  │     │ ChromaDB  │
                              │ LLM +    │◀────│ RAG       │
                              │ Embeddings│    │ Vectors   │
                              └──────────┘     └───────────┘
```

### Infraestrutura LLM/RAG com Docker

A stack de LLM e RAG roda em um Docker Compose separado para evitar impacto direto na stack principal. Com isso, é possível subir apenas a aplicação principal ou adicionar Ollama e ChromaDB quando os endpoints de explicação e chat forem necessários.

Subir apenas Ollama e ChromaDB:

```bash
docker compose -f docker-compose.llm.yaml up -d
```

Verificar o download inicial dos modelos:

```bash
docker logs -f ollama-pull
```

Subir a stack principal junto com LLM/RAG:

```bash
docker compose -f docker-compose.yaml -f docker-compose.llm.yaml up -d
```

Serviços do `docker-compose.llm.yaml`:

| Serviço | Imagem | Porta | Função |
|---|---|---:|---|
| `ollama` | `ollama/ollama:latest` | `11434` | Servidor de LLM e embeddings |
| `ollama-pull` | `ollama/ollama:latest` | - | Init container responsável por baixar os modelos |
| `chromadb` | `chromadb/chroma:latest` | `8100` | Vector store utilizado pelo RAG |

## Containerização e Orquestração

O projeto utiliza Docker para padronizar o ambiente de execução e Docker Compose para subir uma stack integrada com componentes de orquestração, serving, rastreabilidade e monitoramento.

### Papel do Dockerfile

O `Dockerfile` é responsável por empacotar a aplicação Python em uma imagem única, contendo:

- código-fonte do projeto
- dependências do ambiente
- configuração de execução
- ponto de entrada da aplicação

Isso garante reprodutibilidade entre ambientes locais, testes e deploy.

### Papel do Docker Compose

O `docker-compose.yaml` orquestra os serviços do projeto, permitindo subir a stack completa com um único comando.

Na configuração proposta para o ambiente integrado, a stack inclui:

- Airflow 3 para orquestração
- FastAPI para serving do modelo
- MLflow UI para tracking e Model Registry
- Prometheus para coleta de métricas
- Grafana para visualização e monitoramento

### Preparação do ambiente

Antes de subir os containers, é necessário criar a estrutura de pastas usada pela stack e garantir permissões adequadas para escrita de logs, dados, artefatos e configurações.

No terminal, execute:

```bash
# 1. Pastas do Airflow e API
mkdir -p dags logs config plugins fast_api prometheus

# 2. Pastas do Projeto ML (Onde o pipeline grava os resultados)
mkdir -p data/raw data/processed models reports artifacts/plots mlartifacts chroma_data ollama_data

# 3. Estrutura de Provisioning do Grafana (Monitoramento automático)
mkdir -p grafana/provisioning/dashboards grafana/provisioning/datasources

# 4. Cria o arquivo do banco de dados (evita que o Docker crie uma pasta no lugar)
touch mlflow.db

# 5. Permissões de escrita (Crucial para Linux/WSL2)
chmod -R 777 logs dags config plugins data models reports artifacts mlartifacts mlflow.db chroma_data ollama_data
chmod -R 755 grafana/provisioning
```

> **Nota para Linux/WSL2:** caso o Grafana apresente erro de leitura no boot, execute:
> `sudo chown -R 472:0 grafana/provisioning`

### Como iniciar a infraestrutura

1. Certifique-se de que o arquivo `.env` está na raiz do projeto.
2. Ajuste o `AIRFLOW_UID` com o valor retornado por `id -u`, quando aplicável.
3. Garanta que `datasource.yaml`, `dashboards.yaml` e o dashboard exportado em `.json` estejam dentro de `grafana/provisioning/`.
4. Suba a infraestrutura:

```bash
docker compose up -d
```

Para reconstruir as imagens ao subir os serviços:

```bash
docker compose up -d --build
```

Para derrubar a infraestrutura:

```bash
docker compose down
```

### Serviços, portas e acessos

A stack do projeto é composta por serviços que atuam de forma integrada para orquestração, inferência, rastreabilidade e monitoramento.

| Serviço | Função | Porta/URL de referência | Credenciais |
|---|---|---|---|
| Airflow 3 | Orquestração dos fluxos e pipelines | `http://localhost:8080` | `admin / admin_pass` *(ver `.env`)* |
| MLflow UI | Tracking de experimentos e Model Registry | `http://localhost:5001` | - |
| FastAPI | Expor endpoints de inferência, explicação e chat | `http://localhost:8000` | - |
| Grafana | Visualização de métricas e dashboards | `http://localhost:3000` | `admin / grafana_pass` |
| Prometheus | Monitoramento e coleta de métricas | `http://localhost:9090` | - |
| Ollama | Servidor local de LLM e embeddings | `http://localhost:11434` | - |
| ChromaDB | Vector store utilizado pelo RAG | `http://localhost:8100` | - |

> Ajuste essa tabela conforme a configuração final definida no `docker-compose.yaml` do projeto.

### Relação entre Dockerfile e Docker Compose

A explicação do `Dockerfile` deve aparecer junto da seção de containerização porque ele não atua de forma isolada dentro da arquitetura do projeto.

- O **Dockerfile** define como a imagem da aplicação é construída.
- O **Docker Compose** define como os serviços são levantados, conectados e executados em conjunto.

Em outras palavras:

- o `Dockerfile` empacota a aplicação
- o `docker-compose.yaml` organiza a execução da stack completa

Essa combinação é importante porque permite que a aplicação, o MLflow e os componentes de monitoramento sejam executados de forma padronizada, integrada e reproduzível.

### Notas de arquitetura da stack

- **Persistência:** o pipeline de treino utiliza caminhos absolutos como `/opt/airflow/project` para que modelos, artefatos e logs do MLflow fiquem persistidos em volumes locais.
- **Provisioning:** o Grafana pode carregar automaticamente o data source do Prometheus e dashboards salvos em `grafana/provisioning`, evitando configuração manual após o boot.
- **Boot inicial:** o primeiro carregamento da infraestrutura pode levar alguns minutos até que o serviço `airflow-init` finalize as migrações do banco e prepare o ambiente.
- **IA & RAG:** O projeto utiliza Llama 3.2 (via Ollama) e ChromaDB para fornecer uma interface de chat inteligente sobre os dados e modelos via endpoint /chat.


## Próximos passos recomendados

Algumas evoluções naturais para o projeto são:

- promover a versão validada do modelo para o alias `champion` no MLflow Model Registry
- comparar diferentes janelas temporais e conjuntos de features
- testar modelos adicionais, como GRU, TCN ou abordagens baseadas em boosting, para benchmark
- expandir o RAG com dados externos, como notícias e documentos financeiros
- adicionar análise de sentimento como feature complementar para o modelo de previsão
- incluir os resultados quantitativos finais no README após o fechamento dos experimentos
