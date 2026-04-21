FROM apache/airflow:3.2.0

# Usuário root apenas para preparar diretórios se necessário
USER root
RUN mkdir -p /opt/airflow/project && chown -R airflow: /opt/airflow/project

USER airflow

# Copia o arquivo de dependências
COPY --chown=airflow:0 pyproject.toml /opt/airflow/project/pyproject.toml

# Instala as dependências listadas no seu TOML
# O --no-cache-dir é importante para diminuir o tamanho da imagem final
RUN pip install --no-cache-dir "/opt/airflow/project/."

# Define o diretório de trabalho
WORKDIR /opt/airflow/project