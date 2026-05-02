# Arquitetura LLM + RAG + MCP

## Diagrama geral

```mermaid
graph TB
    subgraph Cliente
        CC[Claude Code / MCP Client]
    end

    subgraph MCP["MCP Server (stdio)"]
        T1[predict]
        T2[model_info]
        T3[explain_prediction]
        T4[ask_about_model]
    end

    subgraph API["FastAPI"]
        EP1["POST /predict"]
        EP2["GET /model-info"]
        EP3["POST /explain"]
        EP4["POST /chat"]
    end

    subgraph Services["Camada de Servicos"]
        SVC[service.py<br/>predict_from_close]
        LLM_SVC[llm_service.py<br/>explain_prediction]
        RAG_SVC[rag_service.py<br/>chat_about_model]
    end

    subgraph Infra["Infraestrutura Docker"]
        OL[Ollama<br/>LLM + Embeddings<br/>porta 11434]
        CH[ChromaDB<br/>Vector Store<br/>porta 8100]
        ML[MLflow<br/>Model Registry<br/>porta 5001]
    end

    CC -->|MCP stdio| T1
    CC -->|MCP stdio| T2
    CC -->|MCP stdio| T3
    CC -->|MCP stdio| T4

    T1 -->|HTTP| EP1
    T2 -->|HTTP| EP2
    T3 -->|HTTP| EP3
    T4 -->|HTTP| EP4

    EP1 --> SVC
    EP2 --> SVC
    EP3 --> SVC
    EP3 --> LLM_SVC
    EP4 --> RAG_SVC

    SVC --> ML
    LLM_SVC -->|"POST /api/generate"| OL
    RAG_SVC -->|"POST /api/embeddings"| OL
    RAG_SVC -->|query/add| CH
    RAG_SVC -->|gerar resposta| LLM_SVC
```

## Fluxo do endpoint /explain

```mermaid
sequenceDiagram
    participant U as Usuario/MCP
    participant API as FastAPI
    participant SVC as service.py
    participant LLM as llm_service.py
    participant OL as Ollama

    U->>API: POST /explain {close: 95.5}
    API->>SVC: predict_from_close(95.5)
    SVC-->>API: {predicted_price, predicted_return}
    API->>LLM: explain_prediction(close, price, return)
    LLM->>OL: POST /api/generate {prompt, model}
    OL-->>LLM: {response: "texto explicativo"}
    LLM-->>API: "texto explicativo"
    API-->>U: {explanation, close, predicted_price, predicted_return}
```

## Fluxo do endpoint /chat (RAG)

```mermaid
sequenceDiagram
    participant U as Usuario/MCP
    participant API as FastAPI
    participant RAG as rag_service.py
    participant CH as ChromaDB
    participant OL as Ollama
    participant LLM as llm_service.py

    U->>API: POST /chat {question: "Qual seq_length?"}
    API->>RAG: chat_about_model(question)
    RAG->>CH: query(question, n=3)
    Note over CH,OL: ChromaDB usa Ollama<br/>para gerar embeddings da query
    CH->>OL: POST /api/embeddings
    OL-->>CH: embedding vector
    CH-->>RAG: [chunk1, chunk2, chunk3]
    RAG->>LLM: generate_text(contexto + pergunta)
    LLM->>OL: POST /api/generate
    OL-->>LLM: resposta em linguagem natural
    LLM-->>RAG: resposta
    RAG-->>API: resposta
    API-->>U: {answer, question}
```

## Fluxo do MCP Server

```mermaid
sequenceDiagram
    participant CC as Claude Code
    participant MCP as MCP Server
    participant API as FastAPI

    CC->>MCP: list_tools()
    MCP-->>CC: [predict, model_info, explain_prediction, ask_about_model]

    CC->>MCP: call_tool("predict", {close: 95.5})
    MCP->>API: POST /predict {close: 95.5}
    API-->>MCP: {predicted_price: 95.9, ...}
    MCP-->>CC: TextContent(JSON formatado)
```

## Componentes Docker

Todos os serviços abaixo fazem parte do mesmo `docker-compose.yaml`.

| Servico | Imagem | Porta |
|---------|--------|-------|
| Ollama | ollama/ollama:latest | 11434 |
| ChromaDB | chromadb/chroma:latest | 8100 |
| FastAPI | python:3.12-slim | 8000 |
| MLflow | ghcr.io/mlflow/mlflow:latest | 5001 |
| Prometheus | prom/prometheus:latest | 9090 |
| Grafana | grafana/grafana:latest | 3000 |

## Como rodar

```bash
# Sobe a stack inteira (Airflow + FastAPI + MLflow + Prometheus + Grafana + Ollama + ChromaDB)
docker compose up -d

# Aguardar pull dos modelos do Ollama (~5 min na primeira vez)
docker logs -f ollama-pull
```
