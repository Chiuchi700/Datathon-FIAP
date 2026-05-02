from datetime import datetime, timedelta
from airflow.decorators import dag, task
import sys
import os

# Garante que o Python encontre o pacote 'src' na raiz do projeto
PROJECT_ROOT = "/opt/airflow/project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Argumentos padrão
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="treinamento_lstm_nike_modular",
    default_args=default_args,
    description="Pipeline MLOps Nike: Ingestão -> Processamento -> Treino -> Registro",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["datathon", "pytorch", "mlflow", "nke"],
)
def training_pipeline_mod():

    @task(task_id="1_ingestao_dados_brutos")
    def ingestion():
        """Baixa dados do yfinance e salva o CSV raw."""
        from src.data_loader import download_nike_data, save_raw_data
        from src.main import load_params

        params = load_params()
        logger_name = "airflow_ingestion"

        print(f"Baixando dados para o ticker: {params['data']['ticker']}")
        df, _ = download_nike_data(
            ticker=params["data"]["ticker"], months=params["data"]["months"]
        )

        save_raw_data(df, params["data"]["raw_path"])
        return f"Ingestão concluída: {len(df)} linhas baixadas."

    @task(task_id="2_pre_processamento")
    def preprocessing(ingestion_msg):
        """Lê o raw, aplica transformações e salva o processed."""
        from src.data_loader import load_raw_data
        from src.preprocessing import create_processed_data, save_processed_data
        from src.main import load_params

        print(ingestion_msg)
        params = load_params()

        df_raw = load_raw_data(params["data"]["raw_path"])
        df_proc = create_processed_data(df_raw)

        save_processed_data(df_proc, params["data"]["processed_path"])
        return f"Processamento concluído: Shape final {df_proc.shape}"

    @task(task_id="3_treinamento_modelo")
    def training(prep_msg):
        """Executa o loop de treino do PyTorch e loga métricas no MLflow."""
        from src.main import main as run_training_logic

        print(prep_msg)
        # Chamamos a lógica principal de treino
        run_training_logic()
        return "Treinamento e Log de métricas finalizados."

    @task(task_id="4_verificacao_mlflow")
    def post_check(train_msg):
        """Apenas uma tarefa de fechamento para conferência de status."""
        print(train_msg)
        print("Pipeline finalizado com sucesso!")
        print("Acesse o MLflow na porta 5001 para ver os resultados.")

    # Definindo a jornada dos dados (A -> B -> C -> D)
    msg_ingest = ingestion()
    msg_prep = preprocessing(msg_ingest)
    msg_train = training(msg_prep)
    post_check(msg_train)


# Instancia a DAG
training_pipeline_mod_dag = training_pipeline_mod()
