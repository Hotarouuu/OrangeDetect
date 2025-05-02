import wandb
import os
from dotenv import load_dotenv

load_dotenv()  

dataset_path = os.getenv("DATASET_PATH")

destination_folder = dataset_path

os.makedirs(destination_folder, exist_ok=True)

api = wandb.Api()

dataset = api.artifact("mymlworkspace/wandb-registry-dataset/Images from OrangeDetect:v0")

dataset.download(root=destination_folder)  # Especifica o diretório de destino

