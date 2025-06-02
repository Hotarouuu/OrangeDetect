# Script for testing the model on the test dataset

from src import ViT_Model
from src import dataload
import warnings
import torch
warnings.filterwarnings("ignore")
import argparse
import wandb
from dotenv import load_dotenv
import os
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()  

dataset_path = os.getenv("DATASET_PATH")
model_path = os.getenv("MODELS_FOLDER")
dataset = os.path.join(dataset_path, "processed")

def str2bool(v):
    return v.lower() in ("true", "1", "yes", "y")

parser = argparse.ArgumentParser()
parser.add_argument('--NAME', type=str, help='Experiment name')
parser.add_argument('--TRACKING', type=str2bool, default=False, help='True/False to enable tracking')

args = parser.parse_args()

name = args.NAME
tracking = args.TRACKING

if tracking not in [True, False]:
    raise TypeError('To activate Tracking you must only set True or False')

# Fixing the model name
name = name.replace(" ", "_")

def main():
    # Initialize model and load weights
    model, criterion, _, tokenizer = ViT_Model(num_classes=3)
    best_model = rf"{model_path}\{name}_bestmodel-finetuned.pth"
    
    if not os.path.exists(best_model):
        raise FileNotFoundError(f"Model file not found at {best_model}")
    
    model.load_state_dict(torch.load(best_model, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test dataset
    _, test_dataloader, _ = dataload(dataset)

    # Initialize metrics
    y_correct = []
    y_pred = []

    # Test loop
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            X = tokenizer(X, return_tensors="pt", do_rescale=False).to(device)
            pred = model(**X)
            predicted = pred['logits'].argmax(dim=1)
            y_pred.append(predicted.cpu())
            y_correct.append(y.cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_correct = torch.cat(y_correct, dim=0)

    # Calculate metrics
    f1score = MulticlassF1Score(num_classes=3, average="macro")
    precision_metric = MulticlassPrecision(num_classes=3, average="macro")
    recall_metric = MulticlassRecall(num_classes=3, average="macro")
    confusion_metric = MulticlassConfusionMatrix(num_classes=3)

    f1score.update(y_correct, y_pred)
    precision_metric.update(y_correct, y_pred)
    recall_metric.update(y_correct, y_pred)
    confusion_metric.update(y_correct, y_pred)

    f1sc = f1score.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    confusion = confusion_metric.compute().numpy()

    # Print results
    print(f"\nTest Results:")
    print(f"F1 Score: {f1sc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{confusion}")

    if tracking:
        wandb.init(project="Orange Detect", name=f"test_{name}")
        wandb.log({
            "test_f1": f1sc,
            "test_precision": precision,
            "test_recall": recall,
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=y_pred.numpy(),
                y_true=y_correct.numpy(),
                class_names=[str(i) for i in range(3)]
            )
        })
        artifact = wandb.Artifact(f"final_test_metrics", type="evaluation")
        metrics_file = os.path.join(model_path, f"test_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"F1 Score: {f1sc}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Confusion Matrix:\n{confusion}\n")
        artifact.add_file(metrics_file)
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    main()