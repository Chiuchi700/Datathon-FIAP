# Projeto Integrado: Airflow 3.2.0, FastAPI & Observabilidade

Este ambiente Docker levanta uma infraestrutura completa para orquestração de dados, serviço de API e monitoramento.

## 🛠️ Preparação do Ambiente

Antes de rodar o Docker, crie as pastas necessárias:

mkdir -p dags logs config plugins fast_api prometheus

E garanta as permissões de escrita:

chmod -R 777 logs dags config plugins

## 🚀 Como Iniciar

1. Certifique-se de que o arquivo .env está na raiz.
2. Execute o comando:
   docker compose up -d

## 🔗 Portas e Links de Acesso

- Airflow: http://localhost:8080 (Login: airflow / Senha: airflow)
- FastAPI: http://localhost:8000
- Métricas: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (Login: admin / Senha: admin)

## 📝 Notas
- O Airflow 3.2.0 utiliza o api-server como componente principal.
- O primeiro boot pode levar cerca de 1 minuto para estabilizar o banco de dados.
