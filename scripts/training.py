from src.models.resnet_finetuned import create_resnet50_model
from src.data.dataloader import dataload
from src.utils.logger import init_experiment
from src.test.test import testing_model
from src.train.train import train_model
import mlflow
import mlflow.pytorch
import os
import datetime

import warnings
warnings.filterwarnings("ignore")

def main():

    path_data = r'C:\Users\Lucas\Documents\GitHub\OrangeDetect\data\processed'
    path_ml = r'C:\Users\Lucas\Documents\GitHub\OrangeDetect\experiments'
    name_ml = 'Orange Detect'

    init_experiment(name_ml, path_ml)

    model, criterion, optimizer_conv = create_resnet50_model(num_classes=3)

    train_dataloader, test_dataloader, eval_dataloader = dataload(path_data)

    # Iniciar o run no MLflow
    with mlflow.start_run() as run:

        model.to(device='cuda')

        # Logar hiperparâmetros
        params = {
            'Loss Function' : criterion,
            'Optimizer' : optimizer_conv
        }
        mlflow.log_params(params)

        # Treinamento

        train_model(model, train_dataloader, eval_dataloader, criterion, optimizer_conv, device='cuda', epochs = 5)

        # Testando modelo 

        f1sc, precision, recall, confusion = testing_model(model, test_dataloader, device='cuda')

        # Logar métricas
        mlflow.log_metric("F1 Score", f1sc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        print('\nMétricas logadas')


        # Logando os pesos do modelo
        #model_path = 'resnet-50-finetuned.pth'
        #torch.save(model.state_dict(), model_path)  # Salva só os pesos
        #mlflow.log_artifact(model_path)  # Loga o arquivo

        # Logando o modelo

        mlflow.pytorch.log_model(model, "resnet-50-finetuned")

        # Salvando o modelo

        ## Gera nome automático
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = r"C:\Users\Lucas\Documents\GitHub\OrangeDetect\models"
        save_path = os.path.join(base_path, f"resnet50_{now}")

        mlflow.pytorch.save_model(model, path=save_path)

if __name__ == "__main__":
    main()