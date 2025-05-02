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

load_dotenv()  

dataset_path = os.getenv("DATASET_PATH") # Define variables with .env or add path here
model_path = os.getenv("MODELS_FOLDER") # Define variables with .env or add path here

# Hyperparameters


parser = argparse.ArgumentParser()
parser.add_argument('--LEARNING_RATE', type=float, help='Initial Learning Rate')
parser.add_argument('--EPOCHS', type=int, help='Training Epochs')
parser.add_argument('--NAME', type=str, help='Experiment name')
parser.add_argument('--MODEL', type=str, help='Available models: Resnet50, Resnet101')


args = parser.parse_args()

lr = args.LEARNING_RATE
epochs = args.EPOCHS
name = args.NAME
model_name = args.MODEL

def main():


    if args.LEARNING_RATE is None:
        lr = 0.001
    else:
        pass  

    if model_name  == None:
        raise TypeError('Available models: Resnet50, Resnet101')
    elif model_name == 'resnet101' or model_name == 'Resnet101':
        model, criterion, optimizer_conv = create_resnet101_model(num_classes=3, lr=lr)
    else:
        model, criterion, optimizer_conv = create_resnet50_model(num_classes=3, lr=lr)


    train_dataloader, test_dataloader, eval_dataloader = dataload(dataset_path)

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
    run_name=name
)

    training.train()


    print('\nMetrics logged')


    # Loggin Model Artifact

    torch.save(model.state_dict(), rf"{model_path}\{model_name}-finetuned.pth")

    artifact2 = wandb.Artifact(f'resnet-finetuned', type='model')
    artifact2.add_file(rf"{model_path}\{model_name}-finetuned.pth")
    wandb.log_artifact(artifact2)



if __name__ == "__main__":
    main()