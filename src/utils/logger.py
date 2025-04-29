
import mlflow
import os


def init_experiment(name, path):
    # Corrigido o caminho para o formato de URI 'file:///'
    tracking_uri = f"file:///{os.path.abspath(path)}"
    model_registry_uri = f"{tracking_uri}/models"

    # Configura o URI de rastreamento do MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(model_registry_uri)

    # Cria o experimento
    mlflow.set_experiment(name)

    print('Experiment created.')

