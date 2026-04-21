# Projeto Integrado: Airflow 3, FastAPI & MLOps (MLflow + Grafana)

Este ambiente Docker levanta uma infraestrutura completa para orquestração de dados, treinamento de modelos LSTM, serviço de API e monitoramento.

## 🛠️ Preparação do Ambiente

Antes de rodar o Docker, é necessário criar a estrutura de pastas e garantir as permissões de escrita para que os containers consigam gravar logs, dados e modelos.

No terminal, execute:

# 1. Pastas do Airflow e API
mkdir -p dags logs config plugins fast_api prometheus

# 2. Pastas do Projeto ML (Onde o pipeline grava os resultados)
mkdir -p data/raw data/processed models reports artifacts/plots mlartifacts

# 3. Estrutura de Provisioning do Grafana (Monitoramento automático)
mkdir -p grafana/provisioning/dashboards grafana/provisioning/datasources

# 4. Permissões de escrita (Crucial para Linux/WSL2)
chmod -R 777 logs dags config plugins data models reports artifacts mlartifacts
chmod -R 755 grafana/provisioning

> **Nota para Linux/WSL2:** Caso o Grafana apresente erro de leitura no boot, execute:  
> `sudo chown -R 472:0 grafana/provisioning`

## 🚀 Como Iniciar

1. Certifique-se de que o arquivo `.env` está na raiz do projeto (ajuste o `AIRFLOW_UID` com o comando `id -u`).
2. Certifique-se de que os arquivos `datasource.yaml`, `dashboards.yaml` e o seu arquivo `.json` exportado estão dentro da pasta `grafana/provisioning/`.
3. Suba a infraestrutura:
   docker compose up -d

## 🔗 Portas e Links de Acesso

| Serviço | URL | Credenciais |
| :--- | :--- | :--- |
| **Airflow 3** | http://localhost:8080 | admin / admin_pass (ver .env) |
| **MLflow UI** | http://localhost:5001 | - |
| **FastAPI** | http://localhost:8000 | - |
| **Grafana** | http://localhost:3000 | admin / grafana_pass |
| **Prometheus** | http://localhost:9090 | - |

## 📝 Notas de Arquitetura
- **Persistência:** O pipeline de treino utiliza caminhos absolutos (`/opt/airflow/project`) para garantir que os modelos e logs do MLflow fiquem salvos na sua pasta local.
- **Provisioning:** O Grafana carrega automaticamente o Data Source do Prometheus e os Dashboards salvos na pasta `grafana/provisioning` sem necessidade de configuração manual.
- **Boot:** O primeiro carregamento pode levar cerca de 1 a 2 minutos para que o `airflow-init` finalize as migrações do banco de dados.