from src import create_resnet101_model, create_resnet50_model
from src import dataload
from src import Trainer
import warnings
import torch
warnings.filterwarnings("ignore")
import argparse
import wandb
from dotenv import load_dotenv
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()  

dataset_path = os.getenv("DATASET_PATH") # Define variables with .env or add path here
model_path = os.getenv("MODELS_FOLDER") # Define variables with .env or add path here
dataset = os.path.join(dataset_path, "processed")

# Hyperparameters

def str2bool(v):
    return v.lower() in ("true", "1", "yes", "y")

parser = argparse.ArgumentParser()
parser.add_argument('--LEARNING_RATE', type=float, help='Initial Learning Rate', default=0.001)
parser.add_argument('--EPOCHS', type=int, help='Training Epochs')
parser.add_argument('--NAME', type=str, help='Experiment name')
parser.add_argument('--MODEL', type=str, help='Available models: Resnet50, Resnet101')
parser.add_argument('--TRACKING', type=str2bool, default=False, help='True/False to enable tracking')

args = parser.parse_args()

lr = args.LEARNING_RATE
epochs = args.EPOCHS
name = args.NAME
model_name = args.MODEL
tracking = args.TRACKING

if tracking not in [True, False]:
    raise TypeError('To activate Tracking you must only set True or False')

# Fixing the model name

name = name.replace(" ", "_")

def main():


    if model_name  == None:
        raise TypeError('Available models: Resnet50, Resnet101')
    elif model_name == 'resnet101' or model_name == 'Resnet101':
        model, criterion, optimizer_conv = create_resnet101_model(num_classes=3, lr=lr)
    else:
        model, criterion, optimizer_conv = create_resnet50_model(num_classes=3, lr=lr)


    train_dataloader, test_dataloader, eval_dataloader = dataload(dataset)

    # Training


    config = {
        'Optimizer' : optimizer_conv,
        'Loss Function' : criterion,
        'Model' : model_name,
        'Num Classes' : 3
    }

    training = Trainer(
    model=model,
    train_loader=train_dataloader,
    eval_loader=eval_dataloader,
    test_dataloader=test_dataloader,
    criterion=criterion,
    optimizer=optimizer_conv,
    device='cuda',
    epochs=epochs,
    project="Orange Detect",
    config = config,
    run_name=name,
    tracking=tracking
)

    training.train()

    best_model = rf"{model_path}\{name}_bestmodel-finetuned.pth"
    training.test(best_model)

    # Loggin Model Artifact


    artifact2 = wandb.Artifact(f'resnet-finetuned', type='model')
    artifact2.add_file(best_model)
    wandb.log_artifact(artifact2)



if __name__ == "__main__":
    main()