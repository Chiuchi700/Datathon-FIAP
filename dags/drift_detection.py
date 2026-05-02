from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
import logging

# Configurações padrão
default_args = {
    "owner": "airflow",
    "retries": 1,
}


@dag(
    dag_id="model_drift_monitoring_dag",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["monitoring", "mlops"],
)
def drift_monitoring_pipeline():

    @task.short_circuit(task_id="check_drift_task")
    def detect_drift_logic():
        """
        Lógica para verificar drift.
        Retorna True para seguir o fluxo (retreinar)
        ou False para parar a DAG aqui.
        """
        logging.info("Iniciando verificação de drift nos dados...")

        # Aqui entraria sua lógica real (Ex: Evidently, KS Test, etc)
        # 1. Carregar dados de referência e atuais
        # 2. Calcular métricas de drift
        drift_detected = False

        if drift_detected:
            logging.warning("DRIFT DETECTADO! Prosseguindo para retreinamento.")
        else:
            logging.info("Nenhum drift significativo detectado. Encerrando execução.")

        return drift_detected

    # Como o TriggerDagRunOperator ainda é um operador clássico,
    # instanciamos ele normalmente
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training_pipeline",
        trigger_dag_id="treinamento_lstm_nike_modular",
        wait_for_completion=False,
    )

    # Definindo a dependência
    detect_drift_logic() >> trigger_training


# Instanciando a DAG
drift_monitoring_pipeline()
