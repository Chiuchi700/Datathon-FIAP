from datetime import datetime, timedelta
from airflow.decorators import dag, task
import sys

# Garante que o Python encontre o pacote 'src' na raiz do projeto
sys.path.insert(0, "/opt/airflow/project")

# Agora os imports funcionam de forma limpa
from src.prepare_data import main as run_preparation
from src.main import main as run_training

# Argumentos padrão para controle de retentativas e monitoramento
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="treinamento_lstm_nike",
    default_args=default_args,
    description="Pipeline completa: Download, Processamento e Treino LSTM com MLflow",
    schedule=None,  # Alterado de schedule_interval para schedule
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["datathon", "pytorch", "mlflow", "nke"],
)
def training_pipeline():

    @task(task_id="preparacao_dos_dados")
    def preparation():
        """
        Executa o download via yfinance e salva os CSVs
        nas pastas data/raw e data/processed.
        """
        run_preparation()
        return "Dados preparados com sucesso"

    @task(task_id="treinamento_e_mlflow")
    def training(prep_status):
        """
        Lê os dados processados, treina o modelo PyTorch,
        gera os gráficos e registra tudo no MLflow local.
        """
        print(f"Status da etapa anterior: {prep_status}")
        run_training()
        return "Modelo treinado e registrado"

    # Define o fluxo:  Preparação -> Treinamento
    status_prep = preparation()
    training(status_prep)


# Instancia a DAG para o Airflow detectar
training_pipeline_dag = training_pipeline()
