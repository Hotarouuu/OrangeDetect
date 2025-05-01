from src.models.resnet_finetuned import create_resnet50_model
from src.models.resnet101_finetuned import create_resnet101_model
from src.data.dataloader import dataload
from src.train.train import Trainer
import warnings
import torch
warnings.filterwarnings("ignore")
import argparse
import wandb

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

    path_data = r'C:\Users\Lucas\Documents\GitHub\OrangeDetect\data\processed'
    name_ml = 'Orange Detect'

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


    train_dataloader, test_dataloader, eval_dataloader = dataload(path_data)

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
    dummy_input = torch.randn(1, 3, 224, 224).float().to(device='cuda')
    torch.onnx.export(model, dummy_input, rf"C:\Users\Lucas\Documents\GitHub\OrangeDetect\models\{model_name}-finetuned.onnx")
    torch.save(model.state_dict(), rf"C:\Users\Lucas\Documents\GitHub\OrangeDetect\models\{model_name}-finetuned.pth")

    artifact = wandb.Artifact(f"resnet-finetuned.onnx", type="model")
    artifact2 = wandb.Artifact(f'resnet-finetuned', type='model')
    artifact.add_file(rf"C:\Users\Lucas\Documents\GitHub\OrangeDetect\models\{model_name}-finetuned.onnx")
    artifact2.add_file(rf"C:\Users\Lucas\Documents\GitHub\OrangeDetect\models\{model_name}-finetuned.pth")
    wandb.log_artifact(artifact)
    wandb.log_artifact(artifact2)



if __name__ == "__main__":
    main()