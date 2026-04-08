# Datathon — Previsão do próximo fechamento da Nike com LSTM em PyTorch

## Visão geral

Este projeto foi desenvolvido para o Datathon da pós-graduação em Engenharia de Machine Learning, com foco na previsão do comportamento do ativo **Nike (NKE)**. A solução foi evoluída para um pipeline mais próximo de um cenário de produção, com:

- coleta automática de dados históricos via `yfinance`
- separação entre dados **raw** e **processed**
- engenharia de atributos para séries temporais
- treinamento de modelo **LSTM em PyTorch**
- avaliação com métricas de regressão e comparação com **baseline ingênuo**
- geração de gráficos e logs de execução
- rastreabilidade com **MLflow Tracking**
- versionamento do modelo com **MLflow Model Registry**
- inferência carregando modelo e metadados diretamente do registry

O projeto deixou de depender de artefatos locais fixos para inferência. No estado atual, o modelo treinado e os metadados de pré-processamento são registrados no MLflow e recuperados pelo alias configurado.

---

## Objetivo do projeto

O objetivo é construir uma pipeline de previsão para séries temporais financeiras capaz de estimar o **retorno do próximo dia** da ação da Nike e, a partir disso, reconstruir o **preço previsto**.

A estratégia de prever **retorno futuro** em vez de prever diretamente o preço absoluto foi adotada para reduzir suavização excessiva e melhorar a estabilidade do treinamento. Essa lógica está implementada no pipeline principal, que calcula `target_return_1d` como alvo e reconstrói o preço previsto a partir do último fechamento conhecido. fileciteturn1file14 fileciteturn1file11

---

## Escopo implementado até agora

Atualmente o projeto cobre:

- download do histórico da ação `NKE` com janela configurável em meses
- salvamento do dado bruto em `data/raw/nike_raw.csv`
- geração de base tratada em `data/processed/nike_processed.csv`
- criação de features derivadas de preço, média móvel, volatilidade e volume
- montagem de sequências temporais para LSTM
- split temporal em treino, validação e teste
- escalonamento com `MinMaxScaler` ajustado somente no treino
- treinamento de uma LSTM em PyTorch com early stopping e ajuste adaptativo de learning rate
- logging estruturado em console e arquivo
- registro de parâmetros, métricas, gráficos, modelo e metadados no MLflow
- registro de versões no Model Registry com uso de alias
- inferência carregando modelo e metadata do registry
- notebook EDA separado do pipeline operacional

---

## Arquitetura da solução

### 1. Coleta dos dados

A coleta é feita por `download_nike_data`, que busca os dados no Yahoo Finance, ajusta o intervalo temporal e salva o arquivo bruto na pasta `data/raw`. fileciteturn1file9

### 2. Processamento

O processamento padroniza nomes de colunas, trata tipos, ordena por data, remove inconsistências e cria atributos derivados. Entre as principais features atuais estão:

- `close`
- `return_1d`
- `ma_5_ratio`
- `ma_20_ratio`
- `volatility_10`
- `volume_zscore_20`
- `target_return_1d` como alvo

Esse fluxo está concentrado em `create_processed_data`, enquanto `save_processed_data` persiste a base tratada em disco. fileciteturn1file15

### 3. Preparação das sequências

A função `prepare_sequences` transforma a base em janelas temporais, separa treino/validação/teste e devolve também artefatos importantes do pré-processamento, como:

- `scaler`
- `feature_cols`
- `seq_length`

Esses artefatos são fundamentais para a etapa de inferência, pois garantem o mesmo padrão usado no treinamento. O `MinMaxScaler` é ajustado apenas com o conjunto de treino para evitar leakage. fileciteturn1file14

### 4. Modelo

O modelo atual foi migrado de TensorFlow para **PyTorch**. A arquitetura é composta por:

- `LSTM` bidirecional com 64 unidades na primeira camada
- `Dropout(0.15)`
- segunda `LSTM` com 32 unidades
- mais um `Dropout(0.15)`
- camadas densas `32 -> 16 -> 1`

O treinamento usa:

- `HuberLoss`
- `Adam`
- `ReduceLROnPlateau`
- `EarlyStopping` customizado

Tudo isso está implementado no módulo de treino. fileciteturn1file13 fileciteturn1file18

### 5. Avaliação

Depois do treino, o pipeline gera previsões de retorno no conjunto de teste, reconstrói o preço previsto e compara com o preço real e com um baseline ingênuo. As métricas atuais são:

- `RMSE`
- `MAE`
- `MAPE`
- `Direction Accuracy`
- comparação com `naive_rmse` e `naive_mae`

Além disso, o projeto salva gráficos de loss e de previsão versus real. fileciteturn1file11 fileciteturn1file17

### 6. Tracking e Registry

O projeto foi evoluído para usar **MLflow** com backend local em `sqlite:///mlflow.db` por padrão, além de suporte a artifact root local em `file:./mlartifacts`. A configuração central está em `mlflow_utils.py`, que prepara tracking URI, registry URI e experimento. fileciteturn0file1

Ao final do treino, o pipeline:

- registra parâmetros e métricas da execução
- salva gráficos e logs como artefatos
- loga o modelo PyTorch no MLflow
- salva os metadados de pré-processamento como artefato
- registra uma nova versão no **Model Registry**
- define alias para a versão, como `candidate` ou `champion`

Esse fluxo está implementado em `main.py` e `model_registry.py`. fileciteturn0file8 fileciteturn0file6

### 7. Inferência

A inferência foi ajustada para carregar o modelo e os metadados diretamente do **Model Registry**, sem depender de um `scaler.pkl` ou modelo local em uma pasta fixa. A função `predict_next_day`:

- configura as URIs do MLflow
- baixa os metadados da versão apontada por alias
- carrega o modelo registrado
- transforma a janela de entrada com o mesmo scaler do treino
- retorna o **retorno previsto**
- opcionalmente loga também o **preço previsto** quando recebe `last_close`

Esse comportamento já aparece no script de inferência atual. fileciteturn0file9

---

## Estrutura sugerida do projeto

```text
DATATHON/
├── artifacts/
│   └── plots/
│       ├── prediction_vs_real.png
│       └── training_loss.png
├── data/
│   ├── raw/
│   │   └── nike_raw.csv
│   └── processed/
│       └── nike_processed.csv
├── logs/
│   └── pipeline.log
├── mlartifacts/
├── src/
│   ├── data_loader.py
│   ├── logger_config.py
│   ├── mlflow_utils.py
│   ├── model_registry.py
│   ├── preprocessing.py
│   ├── train.py
│   └── inference/
│       └── inference.py
├── 01_eda_nike.ipynb
├── main.py
├── pyproject.toml
├── mlflow.db
└── README.md
```

> Observação: a estrutura acima representa a organização esperada do repositório. Nos arquivos compartilhados nesta conversa, alguns módulos apareceram de forma isolada, mas o código já está escrito considerando o pacote `src`. fileciteturn1file4

---

## Principais decisões técnicas

### Prever retorno em vez de preço absoluto

A modelagem passou a prever `target_return_1d`, e não o valor direto de `close`. Isso melhora a coerência estatística do problema e reduz o risco de previsões excessivamente suavizadas. fileciteturn1file15

### Evitar vazamento de informação

O `MinMaxScaler` é treinado somente com a partição de treino, o que torna a validação e o teste mais confiáveis. fileciteturn1file14

### Separar EDA do pipeline operacional

O notebook `01_eda_nike.ipynb` ficou responsável pela análise exploratória e pelos insights iniciais, enquanto `main.py` ficou responsável por um fluxo mais operacional, com persistência de dados e artefatos. Essa separação já estava descrita no README anterior e continua válida. fileciteturn1file0

### Centralizar inferência no Model Registry

O estado atual do projeto já reflete uma evolução importante: o modelo de inferência não é mais buscado em uma pasta local fixa, mas sim recuperado do MLflow Registry com base em nome e alias do modelo. fileciteturn0file6 fileciteturn0file9

---

## Como executar o projeto

### 1. Criar e ativar ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

O projeto usa `pyproject.toml` como fonte principal de dependências:

```bash
pip install -e ".[dev]"
```

As dependências incluem `torch`, `mlflow`, `yfinance`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` e pacotes de notebook para EDA. fileciteturn0file7

### 3. Executar o pipeline principal

```bash
python main.py
```

Esse comando executa o fluxo completo:

- baixa os dados da Nike
- cria a base processada
- monta as sequências
- treina o modelo
- calcula métricas
- gera gráficos
- registra a execução no MLflow
- registra a versão do modelo no Model Registry

Esse comportamento está descrito no pipeline do `main.py`. fileciteturn0file8

---

## Variáveis de ambiente principais

O projeto utiliza variáveis de ambiente para controlar comportamento de treino, tracking e registry:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT=file:./mlartifacts
MLFLOW_EXPERIMENT_NAME=nike_lstm_forecasting
REGISTER_MODEL=true
REGISTERED_MODEL_NAME=nike_lstm_forecaster
MODEL_ALIAS=candidate
SAVE_PLOTS=true
PLOTS_DIR=artifacts/plots
LOG_LEVEL=INFO
LOG_DIR=logs
```

Os valores acima refletem os defaults já definidos no código. fileciteturn0file1 fileciteturn0file8 fileciteturn0file9

---

## Como visualizar o MLflow

O pipeline já consegue gravar diretamente no backend local `sqlite:///mlflow.db`, mesmo sem depender de um servidor HTTP ativo no momento da execução. Depois, a interface do MLflow pode ser aberta separadamente apontando para o mesmo backend. Essa lógica está documentada no utilitário de configuração do MLflow. fileciteturn0file1

Exemplo de comando para subir a UI localmente:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination file:./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Com a UI ativa, será possível:

- acompanhar runs do experimento
- comparar métricas entre execuções
- visualizar artefatos
- inspecionar versões do modelo registradas
- validar aliases como `candidate` e `champion`

---

## Como executar a inferência

### Inferência pelo script

Para rodar a inferência local usando a última janela disponível no dado bruto processado:

```bash
python -m src.inference.inference
```

O script:

- lê `data/raw/nike_raw.csv`
- recria a base processada
- recupera `feature_cols` e `seq_length` do registry
- monta a última janela
- carrega o modelo registrado
- imprime o retorno previsto e o preço previsto

Esse fluxo está presente no arquivo de inferência. fileciteturn0file9

### Inferência programática

Também é possível usar a função `predict_next_day(data_last_n_days, last_close=None)` em outro módulo ou API. A entrada precisa respeitar exatamente:

- mesmo `seq_length`
- mesma ordem de `feature_cols`
- mesma estrutura usada no treinamento

Caso seja passado apenas um vetor unidimensional, a função tenta reorganizar a entrada; porém, o formato final ainda precisa ser compatível com o scaler treinado. fileciteturn0file9

---

## Saídas esperadas da execução

Ao final do `main.py`, espera-se encontrar:

### Em disco

- `data/raw/nike_raw.csv`
- `data/processed/nike_processed.csv`
- `artifacts/plots/training_loss.png`
- `artifacts/plots/prediction_vs_real.png`
- `logs/pipeline.log`
- `mlflow.db`
- `mlartifacts/` com os artefatos do MLflow

### No MLflow

- parâmetros do treino
- métricas do modelo e do baseline
- gráficos
- logs
- metadados de pré-processamento
- modelo logado no run
- versão registrada no Model Registry
- alias associado à versão do modelo

Importante: o README antigo mencionava `src/models/lstm_model.keras` e `src/models/scaler.pkl`, mas isso não representa mais o fluxo atual do projeto. Agora o armazenamento principal dos artefatos de inferência ocorre via **MLflow + Model Registry**. fileciteturn1file0 fileciteturn0file6 fileciteturn0file8

---

## Logging

O projeto já possui um logger centralizado com:

- saída em console
- persistência em arquivo `logs/pipeline.log`
- nível configurável por variável de ambiente

Isso ajuda tanto na execução local quanto em futuras execuções via Docker ou API. fileciteturn0file3

---

## EDA

O notebook `01_eda_nike.ipynb` foi mantido como etapa separada para análise exploratória. Ele sustenta a leitura de negócio e ajuda a justificar decisões de modelagem, como:

- comportamento histórico do fechamento
- dinâmica dos retornos diários
- médias móveis
- volatilidade
- sinais relacionados a volume
- correlação entre variáveis derivadas

A separação entre notebook exploratório e pipeline principal é uma das melhorias de organização do projeto. fileciteturn1file0

---

## Limitações atuais

Apesar da evolução da pipeline, ainda existem pontos de atenção naturais para séries temporais financeiras:

- previsão de 1 dia à frente continua sendo um problema de alta variabilidade
- o baseline ingênuo ainda é uma referência importante e precisa ser superado com consistência
- o modelo depende da qualidade e da estabilidade dos sinais derivados criados
- ainda não há, neste repositório, uma API final documentada como camada oficial de serving
- o pipeline ainda pode ser expandido com novos experimentos e mais validações temporais

---

## Próximos passos recomendados

Os próximos passos mais naturais para evolução do projeto são:

1. consolidar a camada de serving com FastAPI consumindo o Model Registry
2. documentar exemplos de payload para inferência com 1 valor e com janela completa
3. promover a versão validada para alias `champion`
4. comparar diferentes janelas temporais e conjuntos de features
5. testar modelos adicionais como GRU, TCN ou abordagens baseadas em boosting para benchmark
6. automatizar treino e deploy com Docker
7. incluir resultados quantitativos finais no README após fechamento dos experimentos
