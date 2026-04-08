source venv/Scripts/activate
pip install -e .[dev]

# todo 
# 3. criar os endpoints de inferência para a api 
# 4. versionar os dados com DVC 
# 6. switch para FastAPI 
# Caio - Baseline treinado e métricas reportadas no MLflow 
# Caio - Pipeline versionado (DVC) e reprodutível treinados. (inferência dos últimos dados reais dentro do docker) 
# Caio - Métricas de negócio mapeadas para métricas técnicas - entender este ponto 

# mlflow
# schema registry
# colocar tipo de dado nos métodos pois era algo que o professor havia comentado


d) Ajustar hiperparâmetros

Primeiros testes:

window_size = 30, 60, 90
batch_size = 16, 32
learning_rate = 0.0005, 0.001
units = (64, 32) e (128, 64)

e) Avaliar métricas mais úteis

Além de RMSE e MAE, use:

MAPE
direção do movimento:
acertou se subiu ou caiu?