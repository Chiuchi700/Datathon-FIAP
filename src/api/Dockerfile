FROM python:3.12-slim

WORKDIR /app

# 1. Copia apenas os arquivos de configuração primeiro para cache de camadas
COPY pyproject.toml README.md ./

# 2. Instala o pip atualizado e as dependências do projeto
# O "." indica que ele vai instalar o projeto local definido no pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 3. Copia o restante do código (pasta src completa)
COPY . /app/src

# 4. Define o PYTHONPATH para que os imports funcionem
ENV PYTHONPATH=/app

# 5. Comando para rodar a API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]