
import mlflow
import os


def init_experiment(name):
    try:
        # Usando o diretório atual para salvar os experimentos
        tracking_uri = f"file:///{os.path.abspath(os.getcwd()).replace('\\', '/')}/mlruns"

        # Configura o tracking URI no MLflow
        mlflow.set_tracking_uri(tracking_uri)

        # Cria ou configura o experimento
        mlflow.set_experiment(name)

        print(f"Experiment '{name}' initialized at {tracking_uri}")
    except Exception as e:
        print(f"Error initializing experiment: {e}")
